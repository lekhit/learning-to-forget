# Experiment 2: Predictor Architecture Benchmarking Results

| Architecture | F1 Score | Precision | Recall | Parameters | Train Time (s) | Latency (ms / 1k tokens) |
|---|---|---|---|---|---|---|
| 2-Layer MLP (Baseline) | 0.9275 | 0.9180 | 0.9373 | 4289 | 13.70 | 0.1140 |
| 1D CNN (Temporal locality) | 0.9274 | 0.9173 | 0.9377 | 4877 | 12.90 | 0.3700 |
| Tiny GRU (Recurrent state) | 0.9244 | 0.9122 | 0.9369 | 3393 | 13.54 | 0.2315 |
| Perceptron Branch Predictor | 0.6796 | 0.5208 | 0.9778 | 17 | 14.54 | 0.0109 |
