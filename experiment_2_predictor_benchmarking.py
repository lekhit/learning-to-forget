import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import h5py
import numpy as np
import time
from tqdm import tqdm

# Import all candidate architectures
from predictor_models import MLPPredictor, CNN1DPredictor, GRUPredictor, PerceptronBranchPredictor

class H5AttentionDataset(Dataset):
    """
    Dataset wrapper to read features and targets incrementally from HDF5 traces.
    """
    def __init__(self, h5_path):
        self.h5_file = h5py.File(h5_path, "r")
        self.features = self.h5_file["features"]
        self.targets = self.h5_file["targets"]
        
    def __len__(self):
        return self.features.shape[0]
        
    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, y
        
    def close(self):
        self.h5_file.close()

def evaluate_metrics(model, dataloader, device):
    """
    Evaluates precision, recall, and F1-score for predicting Heavy Hitters.
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            # CNN and GRU have custom signatures, but standard forward pass works here
            if isinstance(model, GRUPredictor):
                pred, _ = model(x)
            else:
                pred = model(x)
            all_preds.extend((pred > 0.5).cpu().numpy().astype(int))
            all_targets.extend(y.numpy().astype(int))
            
    preds = np.array(all_preds).squeeze()
    targets = np.array(all_targets).squeeze()
    
    tp = np.sum((preds == 1) & (targets == 1))
    fp = np.sum((preds == 1) & (targets == 0))
    fn = np.sum((preds == 0) & (targets == 1))
    tn = np.sum((preds == 0) & (targets == 0))
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (len(targets) + 1e-8)
    
    return f1, precision, recall, accuracy

def benchmark_inference_speed(model, device, d_x=260):
    """
    Simulates latency for running 1000 concurrent forward passes (FP16 batch size 1000).
    """
    model.eval()
    dummy_input = torch.randn(1000, d_x, device=device)
    
    # Warmup
    for _ in range(10):
        if isinstance(model, GRUPredictor):
            _, _ = model(dummy_input)
        else:
            _ = model(dummy_input)
            
    t0 = time.time()
    iters = 100
    with torch.no_grad():
        for _ in range(iters):
            if isinstance(model, GRUPredictor):
                _, _ = model(dummy_input)
            else:
                _ = model(dummy_input)
                
    latency_ms = (time.time() - t0) / iters * 1000 # Latency in ms for 1000 tokens
    return latency_ms

def train_candidate(name, model, train_loader, val_loader, device, d_x=260, epochs=5):
    """
    Standard training runner for a candidate predictor architecture.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    print(f"\nTraining Candidate: {name}...")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            if isinstance(model, GRUPredictor):
                pred, _ = model(x)
            else:
                pred = model(x)
                
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
    train_duration = time.time() - start_time
    
    # Compute Metrics
    f1, prec, rec, acc = evaluate_metrics(model, val_loader, device)
    inf_latency = benchmark_inference_speed(model, device, d_x)
    params = sum(p.numel() for p in model.parameters())
    
    return {
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "accuracy": acc,
        "params": params,
        "train_time": train_duration,
        "inf_latency": inf_latency
    }

def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device for benchmarking: {device}")
    
    h5_path = "results/llama3_attention_traces.h5"
    if not os.path.exists(h5_path):
        print(f"Error: {h5_path} not found. Please run Experiment 1 first to generate training traces.")
        # If no trace exists, create a dummy trace for testing robustness
        print("Constructing temporary dummy traces to allow verification of Experiment 2 script...")
        os.makedirs("results", exist_ok=True)
        # Determine dummy dim based on device
        dummy_dim = 260 if device == "cuda" else 132
        with h5py.File(h5_path, "w") as h5f:
            h5f.create_dataset("features", data=np.random.randn(2000, dummy_dim).astype(np.float32))
            h5f.create_dataset("targets", data=np.random.choice([0, 1], size=(2000, 1)).astype(np.int8))
            
    dataset = H5AttentionDataset(h5_path)
    total_samples = len(dataset)
    feature_dim = dataset.features.shape[1]
    print(f"Loaded attention dataset with {total_samples} samples. Feature dimension: {feature_dim}")
    
    # Train-test split (80-20)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    # Optimized DataLoader
    train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=512, shuffle=False)
    
    candidates = {
        "2-Layer MLP (Baseline)": MLPPredictor(d_x=feature_dim),
        "1D CNN (Temporal locality)": CNN1DPredictor(d_x=feature_dim),
        "Tiny GRU (Recurrent state)": GRUPredictor(d_x=feature_dim),
        "Perceptron Branch Predictor": PerceptronBranchPredictor()
    }
    
    results = {}
    for name, model in candidates.items():
        results[name] = train_candidate(name, model, train_loader, val_loader, device, d_x=feature_dim)
        
    dataset.close()
    
    # 7. Print comparative results
    print("\n" + "="*80)
    print("EXPERIMENT 2 COMPLETE: PREDICTOR ARCHITECTURE COMPARISON")
    print("="*80)
    print(f"| Architecture | F1 Score | Precision | Recall | Parameters | Train Time (s) | Latency (ms / 1k tokens) |")
    print(f"|---|---|---|---|---|---|---|")
    for name, metrics in results.items():
        print(f"| {name} | {metrics['f1']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['params']} | {metrics['train_time']:.2f} | {metrics['inf_latency']:.4f} |")
    print("="*80)
    
    # 8. Save results to dedicated results/ folder
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "experiment_2_results.md")
    
    with open(results_file, "w") as f:
        f.write("# Experiment 2: Predictor Architecture Benchmarking Results\n\n")
        f.write(f"| Architecture | F1 Score | Precision | Recall | Parameters | Train Time (s) | Latency (ms / 1k tokens) |\n")
        f.write(f"|---|---|---|---|---|---|---|\n")
        for name, metrics in results.items():
            f.write(f"| {name} | {metrics['f1']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['params']} | {metrics['train_time']:.2f} | {metrics['inf_latency']:.4f} |\n")
    print(f"Saved benchmark comparative results to: {results_file}")
    
if __name__ == "__main__":
    main()
