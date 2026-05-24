import torch
import torch.nn as nn
import numpy as np

class AttentionHookManager:
    """
    Manages registration and execution of PyTorch forward hooks 
    to extract intermediate attention states, query/key/value features,
    and relative token statistics during causal decoding.
    """
    def __init__(self, model, num_layers=32, num_heads=32, num_kv_heads=8, head_dim=128):
        self.model = model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        
        self.hooks = []
        # Structure to hold intermediate activations per layer
        # Layer -> Step -> Head -> Token statistics
        self.captured_attention = [[] for _ in range(num_layers)]
        self.captured_k_states = [[] for _ in range(num_layers)]
        self.captured_v_states = [[] for _ in range(num_layers)]
        
    def register_hooks(self):
        """
        Dynamically registers forward hooks on all attention blocks and their projections.
        Supports standard Hugging Face Llama and Qwen architectures.
        """
        self.clear_cache()
        
        def make_attn_hook(layer_idx):
            def hook_fn(module, input_t, output_t):
                # HF attention output: (attn_output, attn_weights_if_output, past_key_value)
                # We extract the output attention weights if returned
                if isinstance(output_t, tuple) and len(output_t) > 1 and output_t[1] is not None:
                    # Shape: [batch_size, num_heads, q_len, kv_len]
                    self.captured_attention[layer_idx].append(output_t[1].detach().cpu())
                elif not isinstance(output_t, tuple) and hasattr(module, "captured_weights"):
                    # Fallback or direct capturing
                    pass
            return hook_fn

        def make_k_hook(layer_idx):
            def hook_fn(module, input_t, output_t):
                # output_t shape: [batch_size, seq_len, num_kv_heads * head_dim]
                bsz, seq_len, _ = output_t.shape
                k_state = output_t.detach().cpu().view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
                self.captured_k_states[layer_idx].append(k_state)
            return hook_fn

        def make_v_hook(layer_idx):
            def hook_fn(module, input_t, output_t):
                # output_t shape: [batch_size, seq_len, num_kv_heads * head_dim]
                bsz, seq_len, _ = output_t.shape
                v_state = output_t.detach().cpu().view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
                self.captured_v_states[layer_idx].append(v_state)
            return hook_fn

        # Register on Llama/Qwen layers
        layers = []
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            layers = self.model.transformer.h
            
        for idx, layer in enumerate(layers):
            if idx >= self.num_layers:
                break
            
            # Check for standard self_attn module
            self_attn = getattr(layer, "self_attn", getattr(layer, "attn", None))
            if self_attn is not None:
                # 1. Attention weight hook
                attn_hook = self_attn.register_forward_hook(make_attn_hook(idx))
                self.hooks.append(attn_hook)
                
                # 2. Key projection hook
                if hasattr(self_attn, "k_proj"):
                    k_hook = self_attn.k_proj.register_forward_hook(make_k_hook(idx))
                    self.hooks.append(k_hook)
                    
                # 3. Value projection hook
                if hasattr(self_attn, "v_proj"):
                    v_hook = self_attn.v_proj.register_forward_hook(make_v_hook(idx))
                    self.hooks.append(v_hook)
            
        print(f"Successfully registered {len(self.hooks)} forward hooks across model attention subsystems.")

    def remove_hooks(self):
        """
        Removes all active hooks to restore the original forward pass performance.
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        print("All attention hooks successfully removed.")

    def clear_cache(self):
        """
        Clears all intermediate activation memory.
        """
        self.captured_attention = [[] for _ in range(self.num_layers)]
        self.captured_k_states = [[] for _ in range(self.num_layers)]
        self.captured_v_states = [[] for _ in range(self.num_layers)]

    def compile_features(self, layer_idx, head_idx, token_idx, current_step):
        """
        Computes the (2*head_dim + 4)-dimensional feature vector x_{i,h} for a specific token
        in a specific head at the current sequence step.
        
        x = [ LN(k) || LN(v) || log(t-i) || initial_attention || cumulative_attention || std(k) ]
        """
        dim = 2 * self.head_dim + 4
        # Ensure we have captured states
        if len(self.captured_k_states[layer_idx]) == 0:
            return torch.zeros(dim)
            
        # Get latest key and value states
        # Shape: [batch, kv_heads, seq_len, head_dim]
        k_tensor = self.captured_k_states[layer_idx][-1][0, head_idx, token_idx, :] # [head_dim]
        v_tensor = self.captured_v_states[layer_idx][-1][0, head_idx, token_idx, :] # [head_dim]
        
        # Apply simple layer normalization/scaling
        k_norm = (k_tensor - k_tensor.mean()) / (k_tensor.std() + 1e-8)
        v_norm = (v_tensor - v_tensor.mean()) / (v_tensor.std() + 1e-8)
        
        # Scalar features
        rel_pos = np.log(max(current_step - token_idx, 1))
        
        # Attention scores for this token
        # Get attention matrix history
        attn_history = []
        for step_matrix in self.captured_attention[layer_idx]:
            # step_matrix shape: [batch, heads, q_len, kv_len]
            # Extract score for current head and target token
            score = step_matrix[0, head_idx, -1, token_idx].item()
            attn_history.append(score)
            
        # Cumulative attention
        cum_attn = sum(attn_history)
        
        # Initial attention window (average of first 16 steps)
        init_attn = np.mean(attn_history[:16]) if len(attn_history) > 0 else 0.0
        
        # Concatenate into dynamic feature vector
        features = torch.zeros(dim)
        features[:self.head_dim] = k_norm
        features[self.head_dim:2*self.head_dim] = v_norm
        features[2*self.head_dim] = rel_pos
        features[2*self.head_dim + 1] = cum_attn
        features[2*self.head_dim + 2] = init_attn
        features[2*self.head_dim + 3] = k_tensor.std() # Key magnitude
        
        return features
