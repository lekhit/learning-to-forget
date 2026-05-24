import os
import sys
import time
import types
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from predictor_models import PerceptronBranchPredictor

# 1. Hardware Detection
def detect_hardware():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# Standard Rotary Position Embeddings (RoPE) Helpers
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embeddings to the query and key tensors."""
    # Match dimensions based on dim shape (cos/sin are pre-sliced by the caller)
    if cos.dim() == 4:
        # Already has correct number of dimensions
        pass
    elif cos.dim() == 3:
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
    elif cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(0).unsqueeze(unsqueeze_dim)
        
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class ActiveEvictionAttentionPatch:
    """
    Wraps the attention block forward pass to perform active in-place eviction on GPU.
    """
    def __init__(self, original_attention, layer_idx, num_heads, num_kv_heads, head_dim, threshold=0.3):
        self.original_attention = original_attention
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.threshold = threshold
        self.original_forward = None  # Populated during patching
        
        # Initialize register-level branch predictors per head
        self.device = next(original_attention.parameters()).device
        self.predictors = nn.ModuleList([
            PerceptronBranchPredictor().to(self.device) for _ in range(num_kv_heads)
        ])
        
        # Fused SGD optimizers for training the predictors in-place
        self.optimizers = [
            torch.optim.SGD(p.parameters(), lr=0.01) for p in self.predictors
        ]
        
        # Hardware branch prediction initialization (strongly keep)
        for p in self.predictors:
            nn.init.constant_(p.fc[0].bias, 1.5)
            nn.init.normal_(p.fc[0].weight, std=0.01)

    def forward(self, *args, **kwargs):
        """
        Monkey-patched attention forward pass that prunes the cache in-place 
        before calling the highly-optimized original Hugging Face attention pass.
        """
        # 1. Resolve past_key_values cache
        past_key_value = kwargs.get("past_key_value", kwargs.get("past_key_values", None))
        if past_key_value is None and len(args) > 3:
            past_key_value = args[3]
                
                
        # 2. Perform Active Eviction on past_key_value cache (DynamicCache in HF)
        if past_key_value is not None and hasattr(past_key_value, "key_cache") and len(past_key_value.key_cache) > self.layer_idx:
            cached_keys = past_key_value.key_cache[self.layer_idx]
            cached_values = past_key_value.value_cache[self.layer_idx]
            
            # Sequence length of the history stored in cache
            history_len = cached_keys.shape[-2]
            max_budget = 64
            
            if history_len > max_budget:
                # Synchronized across layers: Layer 0 computes the kept indices and stores them on past_key_value
                if self.layer_idx == 0 or not hasattr(past_key_value, "kept_indices"):
                    # Features for history tokens (first 16 dims of keys)
                    features = cached_keys[:, :, :, :16].detach() # [bsz, kv_heads, history_len, 16]
                    
                    # Predict salience for all history tokens
                    with torch.enable_grad():
                        salience_scores = []
                        for h in range(self.num_kv_heads):
                            score = self.predictors[h](features[:, h, :, :]) # [bsz, history_len, 1]
                            salience_scores.append(score)
                        salience_scores = torch.stack(salience_scores, dim=1).squeeze(-1) # [bsz, kv_heads, history_len]
                    
                    # Global top-k index selection averaged across all attention heads to align K and V
                    avg_scores = salience_scores[0].mean(dim=0) # [history_len]
                    middle_avg_scores = avg_scores[4:-16]
                    k_heavy = max_budget - 4 - 16 # Remaining heavy hitter slots
                    
                    if middle_avg_scores.shape[0] > k_heavy:
                        _, topk_indices = torch.topk(middle_avg_scores, k=k_heavy, dim=-1)
                        topk_indices = topk_indices + 4
                        
                        # Construct sorted list of kept indices
                        sinks = list(range(4))
                        recent = list(range(history_len - 16, history_len))
                        kept_indices = sorted(sinks + topk_indices.tolist() + recent)
                    else:
                        kept_indices = list(range(history_len))
                        
                    # Save on the shared past_key_value cache object
                    past_key_value.kept_indices = kept_indices
                else:
                    # Retrieve the pre-computed kept_indices from layer 0
                    kept_indices = past_key_value.kept_indices
                    
                # Physically prune the cache in-place!
                past_key_value.key_cache[self.layer_idx] = cached_keys[:, :, kept_indices, :]
                past_key_value.value_cache[self.layer_idx] = cached_values[:, :, kept_indices, :]
                
                # Physically slice the attention mask to match the new cache size!
                attention_mask = kwargs.get("attention_mask", None)
                mask_in_args = False
                if attention_mask is None and len(args) > 2:
                    attention_mask = args[2]
                    mask_in_args = True
                    
                if attention_mask is not None:
                    # Slice attention mask to match kept history indices + new token index (history_len)
                    target_indices = kept_indices + [history_len]
                    attention_mask = attention_mask[..., target_indices]
                    if mask_in_args:
                        args = list(args)
                        args[2] = attention_mask
                        args = tuple(args)
                    else:
                        kwargs["attention_mask"] = attention_mask
                    
        # Check if output_attentions was requested by the outer caller
        user_output_attentions = kwargs.get("output_attentions", False)
        
        # 3. Call optimized original forward pass to compute attention
        # We pass output_attentions=True to collect weights for active learning
        kwargs["output_attentions"] = True
        outputs = self.original_forward(*args, **kwargs)
        
        # Now, original_forward returned (attn_output, attn_weights)
        attn_output = outputs[0]
        attn_weights = outputs[1]
        
        # 4. In-Place Predictor Backpropagation (Active Learning)
        if past_key_value is not None and hasattr(past_key_value, "key_cache") and len(past_key_value.key_cache) > self.layer_idx:
            if attn_weights is not None:
                curr_history_len = past_key_value.key_cache[self.layer_idx].shape[-2]
                
                with torch.enable_grad():
                    for h in range(self.num_kv_heads):
                        self.optimizers[h].zero_grad()
                        # target shape: [bsz, curr_history_len]
                        q_head_idx = h * (self.num_heads // self.num_kv_heads)
                        target = attn_weights[:, q_head_idx, -1, :curr_history_len].detach()
                        
                        # Compute features and prediction for history tokens
                        features = past_key_value.key_cache[self.layer_idx][:, h, :curr_history_len, :16].detach()
                        pred = self.predictors[h](features).squeeze(-1) # [bsz, curr_history_len]
                        
                        loss = nn.functional.binary_cross_entropy(pred, (target > 0.05).float())
                        loss.backward()
                        self.optimizers[h].step()
                        
            # Cleanup kept_indices on the last layer so it doesn't leak to subsequent token steps!
            if self.layer_idx == len(past_key_value.key_cache) - 1 and hasattr(past_key_value, "kept_indices"):
                delattr(past_key_value, "kept_indices")
                        
        # Construct the return value matching what the caller expected!
        if user_output_attentions:
            return (attn_output, attn_weights)
        else:
            return (attn_output, None)

def apply_eviction_patch(model, threshold=0.3):
    """
    Monkey-patches the attention modules of the causal language model with active eviction layers.
    """
    config = model.config
    num_heads = getattr(config, "num_attention_heads", 12)
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = getattr(config, "hidden_size", 768) // num_heads
    
    patches = []
    # Identify layers depending on model architecture
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    else:
        print("Warning: Unknown model layer structure. Custom patching skipped.")
        return []
        
    for idx, layer in enumerate(layers):
        # Patch Qwen2Attention / LlamaAttention
        original_self_attn = layer.self_attn if hasattr(layer, "self_attn") else layer.attn
        
        # Instantiate patch
        attn_patch = ActiveEvictionAttentionPatch(
            original_attention=original_self_attn,
            layer_idx=idx,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            threshold=threshold
        )
        
        # Save the original forward pass
        attn_patch.original_forward = original_self_attn.forward
        
        # Overwrite the forward pass
        # Use types.MethodType with a dedicated closure to avoid lexical binding issues in lambdas
        def make_patched_forward(patch_instance):
            return lambda self, *args, **kwargs: patch_instance.forward(*args, **kwargs)
            
        original_self_attn.forward = types.MethodType(
            make_patched_forward(attn_patch),
            original_self_attn
        )
        patches.append(attn_patch)
        
    print(f"Patched {len(patches)} attention blocks with GPU-synchronous in-place eviction.")
    return patches

def main():
    device = detect_hardware()
    print(f"Executing active eviction evaluation on: {device}")
    
    # 1. Load Causal Model
    # On CUDA, load 4-bit Llama-3-8B. On Apple Silicon, load native Qwen-0.5B.
    if device == "cuda":
        model_id = "MaziyarPanahi/Meta-Llama-3-8B-Instruct-AWQ"
        is_quantized = True
    else:
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        is_quantized = False
        
    print(f"Loading tokenizer & model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    try:
        if is_quantized and device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                attn_implementation="eager",
                device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device == "mps" else torch.float32,
                attn_implementation="eager",
                device_map="auto" if device == "mps" else None,
                trust_remote_code=True
            )
            if device == "cpu":
                model = model.to(device)
    except Exception as e:
        print(f"Model load failed: {e}. Falling back to CPU-friendly GPT-2...")
        model_id = "distilgpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

    # 2. Patch the model layers
    patches = apply_eviction_patch(model, threshold=0.35)

    # 3. Benchmark Downstream Performance
    prompt = "Translate this text to French: 'Large language models are revolutionizing technology and open new horizons for developers.' Output:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    print("\nStarting autoregressive text generation with active eviction...")
    t0 = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            use_cache=True,
            do_sample=False
        )
        
    duration = time.time() - t0
    num_new_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\n" + "="*80)
    print("EXPERIMENT 3 EVICTION VALIDATION RESULTS")
    print("="*80)
    print(f"Generated text:\n{response}")
    print("-"*80)
    print(f"Generation Latency: {duration:.2f} seconds")
    print(f"Tokens Generated: {num_new_tokens}")
    print(f"Generation Throughput: {num_new_tokens / (duration + 1e-8):.2f} tokens/second (TPS)")
    print("="*80)
    
    # 4. Save results to dedicated results/ folder
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "experiment_3_results.md")
    
    with open(results_file, "w") as f:
        f.write("# Experiment 3: In-Place Eviction Generation Results\n\n")
        f.write(f"**Device Used:** {device}\n\n")
        f.write(f"### Generated Text Output:\n```\n{response}\n```\n\n")
        f.write(f"### Performance Metrics:\n")
        f.write(f"- **Generation Latency:** {duration:.2f} seconds\n")
        f.write(f"- **Tokens Generated:** {num_new_tokens}\n")
        f.write(f"- **Throughput (TPS):** {num_new_tokens / (duration + 1e-8):.2f} tokens/second\n")
    print(f"Saved generation benchmark results to: {results_file}")

if __name__ == "__main__":
    main()
