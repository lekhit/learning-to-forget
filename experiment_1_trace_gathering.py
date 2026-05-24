import os
import sys
import torch
import h5py
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# Import the hook manager we created
from attention_hook import AttentionHookManager

def detect_environment():
    """
    Detects the execution hardware environment and returns optimized configuration.
    """
    if torch.cuda.is_available():
        device = "cuda"
        # On CUDA, use 4-bit Llama-3-8B AWQ
        model_id = os.getenv("MODEL_ID", "MaziyarPanahi/Meta-Llama-3-8B-Instruct-AWQ")
        is_quantized = True
        print(f"CUDA hardware detected. Using production target: {model_id}")
    elif torch.backends.mps.is_available():
        device = "mps"
        # On Apple Silicon macOS, load a tiny Llama-compatible Qwen model to run natively with MPS
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        is_quantized = False
        print(f"Apple Silicon (MPS) detected. Using native local target: {model_id} for fast local testing.")
    else:
        device = "cpu"
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        is_quantized = False
        print(f"CPU hardware detected. Using native CPU target: {model_id}")
        
    return device, model_id, is_quantized

def main():
    device, model_id, is_quantized = detect_environment()
    
    # 1. Output Files
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "llama3_attention_traces.h5")
    
    # Clean old run files if present
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"Removed old trace file: {output_file}")

    # 2. Load Tokenizer & Model
    print(f"Loading Tokenizer for: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"Loading Model for: {model_id} on {device}...")
    try:
        if is_quantized and device == "cuda":
            # Load AWQ quantized model
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                attn_implementation="eager"
            )
        else:
            # Load smaller/fallback model in standard float16 or float32 for CPU/MPS
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device == "mps" else torch.float32,
                device_map="auto" if device == "mps" else None,
                trust_remote_code=True,
                attn_implementation="eager"
            )
            if device == "cpu":
                model = model.to(device)
    except Exception as e:
        print(f"Error loading model {model_id}: {e}")
        print("Falling back to standard CPU-friendly GPT-2 for local test validation...")
        model_id = "distilgpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="eager").to(device)

    # 3. Model Properties Extraction
    config = model.config
    num_layers = getattr(config, "num_hidden_layers", 12)
    num_heads = getattr(config, "num_attention_heads", 12)
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = getattr(config, "hidden_size", 768) // num_heads
    
    print(f"Model properties: Layers={num_layers}, Query Heads={num_heads}, KV Heads={num_kv_heads}, Head Dim={head_dim}")

    # 4. Instantiate and Register Hooks
    hook_manager = AttentionHookManager(
        model=model,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim
    )
    hook_manager.register_hooks()

    # 5. Load Dataset (GSM8k or GovReport for long text reasoning)
    print("Loading text corpus...")
    dataset = []
    try:
        # Try loading govreport
        print("Attempting to load GovReport from HF Hub...")
        ds = load_dataset("govreport", split="test", streaming=True)
        for sample in ds:
            dataset.append(sample.get("report", ""))
            if len(dataset) >= 10:
                break
    except Exception as e:
        print(f"Failed to load GovReport streaming: {e}. Trying wikitext...")
        try:
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", streaming=True)
            for sample in ds:
                dataset.append(sample.get("text", ""))
                if len(dataset) >= 10:
                    break
        except Exception as e2:
            print(f"Failed to load wikitext: {e2}. Falling back to high-quality, local synthetic long-text generator...")
            # Generate highly representative synthetic text containing realistic English vocabulary
            synthetic_topics = [
                "Artificial intelligence and machine learning are transforming the landscape of modern technology. Deep learning models, particularly large language models, use massive transformer architectures to capture long-range contextual relationships in text.",
                "In computer architecture, memory hierarchies are optimized to reduce access latency. CPU branch prediction utilizes perceptrons and history registers to anticipate instruction paths, minimizing execution bubbles and cache misses.",
                "The key-value cache is the primary bottleneck in autoregressive decoding for large models. Compressing and evicting cached states dynamically allows for long-context sequences without incurring out-of-memory errors on local GPUs.",
                "Causal language modeling relies on attention maps where each token attends to previous context. Active online learning predictors can identify heavy hitter tokens that are consistently critical for generating high-quality text."
            ]
            # Multiply to create long context samples
            for i in range(10):
                text_block = " ".join([np.random.choice(synthetic_topics) for _ in range(50)]) # Yields very long contexts (>1500 tokens)
                dataset.append(text_block)
            print(f"Successfully generated {len(dataset)} synthetic long-context validation samples.")


    # 6. Extract Traces & Compile Features into HDF5
    MAX_SAMPLES = 5  # Small sample count for quick local validation
    MAX_LENGTH = 1024  # Capping sequence length for 2080 Ti & Mac compatibility
    FORWARD_WINDOW = 32  # Salience lookahead window
    feature_dim = 2 * head_dim + 4
    print(f"Starting trace gathering. Targets: {MAX_SAMPLES} samples up to {MAX_LENGTH} context length... Feature Dim: {feature_dim}")
    
    with h5py.File(output_file, "w") as h5f:
        # We start with empty datasets and resize incrementally
        x_ds = h5f.create_dataset("features", shape=(0, feature_dim), maxshape=(None, feature_dim), dtype="float32", chunks=(4096, feature_dim))
        y_ds = h5f.create_dataset("targets", shape=(0, 1), maxshape=(None, 1), dtype="int8", chunks=(4096, 1))
        
        sample_count = 0
        pbar = tqdm(total=MAX_SAMPLES, desc="Processing samples")
        
        for sample in dataset:
            if sample_count >= MAX_SAMPLES:
                break
                
            if isinstance(sample, dict):
                text = sample.get("report", sample.get("text", ""))
            else:
                text = sample
            if len(text.strip()) < 100:
                continue  # Skip short lines
                
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            seq_len = inputs["input_ids"].shape[1]
            
            if seq_len < FORWARD_WINDOW + 20:
                continue  # Skip sequences that are too short to calculate lookahead
                
            hook_manager.clear_cache()
            
            # Execute standard causal generation pass
            with torch.no_grad():
                _ = model(**inputs, output_attentions=True)
                
            # Check if attention traces were captured successfully
            if len(hook_manager.captured_attention[0]) > 0:
                # We extract features and salience targets
                features_list = []
                targets_list = []
                
                # Iterate over layers and heads to compile features
                # To avoid massive file sizes during local validation, we sample a subset of layers/heads
                sampled_layers = [0, num_layers // 2, num_layers - 1]
                sampled_heads = list(range(0, num_kv_heads, max(1, num_kv_heads // 4)))
                
                # Causal extraction loop
                # We calculate features for tokens, checking their future salience
                for t in range(20, seq_len - FORWARD_WINDOW):
                    for l in sampled_layers:
                        for h in sampled_heads:
                            # Features for token t at head h, layer l
                            x = hook_manager.compile_features(l, h, t, seq_len)
                            
                            # Target salience based on future window attention
                            # Lookahead attention sum for token t over the next FORWARD_WINDOW steps
                            future_attns = []
                            for step_idx in range(t + 1, min(t + 1 + FORWARD_WINDOW, seq_len)):
                                # If captured_attention has length 1, we are in prefill/parallel mode
                                if len(hook_manager.captured_attention[l]) == 1:
                                    score = hook_manager.captured_attention[l][0][0, h, step_idx, t].item()
                                    future_attns.append(score)
                                # If captured_attention has length > 1, we are in autoregressive mode
                                elif step_idx < len(hook_manager.captured_attention[l]):
                                    score = hook_manager.captured_attention[l][step_idx][0, h, -1, t].item()
                                    future_attns.append(score)
                                    
                            future_sum = sum(future_attns)
                            y = 1 if future_sum > 0.02 else 0
                            
                            features_list.append(x.numpy())
                            targets_list.append([y])
                
                # Append to HDF5 dynamically
                if len(features_list) > 0:
                    features_arr = np.array(features_list, dtype=np.float32)
                    targets_arr = np.array(targets_list, dtype=np.int8)
                    
                    curr_size = x_ds.shape[0]
                    new_size = curr_size + features_arr.shape[0]
                    
                    x_ds.resize(new_size, axis=0)
                    y_ds.resize(new_size, axis=0)
                    
                    x_ds[curr_size:new_size] = features_arr
                    y_ds[curr_size:new_size] = targets_arr
                    
                    sample_count += 1
                    pbar.update(1)
                    print(f"\nSaved {features_arr.shape[0]} feature vectors from sample {sample_count} (Seq Len: {seq_len}). Total vectors: {new_size}")
                    
        pbar.close()

    # 7. Cleanup
    hook_manager.remove_hooks()
    print("==================================================================")
    print(f"EXPERIMENT 1 COMPLETE!")
    print(f"HDF5 dataset successfully constructed at: {output_file}")
    with h5py.File(output_file, "r") as h5f:
        print(f"Total training vectors collected: {h5f['features'].shape[0]}")
        print(f"Features shape: {h5f['features'].shape}, Targets shape: {h5f['targets'].shape}")
    print("==================================================================")

if __name__ == "__main__":
    main()
