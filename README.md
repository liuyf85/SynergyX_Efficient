# SynergyX Optimization: Efficient Drug Synergy Prediction

This repository contains the implementation of **SynergyX**, a multi-modality mutual attention network for interpretable drug synergy prediction. We optimize the SynergyX framework to improve computational efficiency while maintaining high prediction accuracy for large-scale drug synergy prediction. The key optimizations include **FlashAttention**, **mixed-precision training**, and **rewritten LayerNorm**, which lead to significant improvements in training and inference speed, as well as memory consumption.

## Key Features

* **FlashAttention**: Optimizes the attention mechanism by minimizing memory bandwidth overhead, leading to faster computations.
* **Mixed-Precision Training**: Utilizes FP16 precision for most computations, maintaining model accuracy while reducing memory consumption and computation time.
* **Rewritten LayerNorm**: Replaces the custom LayerNorm implementation with PyTorch's built-in `nn.LayerNorm` for faster execution and reduced kernel launch overhead.
* **Efficient Drug Synergy Prediction**: Scalable to large datasets, improving training and inference speed without sacrificing accuracy.

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/liuyf85/SynergyX_Efficient.git
   ```

2. Download the dataset from the [SynergyX](https://github.com/GSanShui/SynergyX) and place it in the `data` folder.

## Usage

### Training the Model

To train the SynergyX model with the optimizations, use the following command:

```bash
sh run.sh
```

### Evaluating Performance

We evaluate the following optimizations:

1. **FlashAttention**: Significant speedup in the attention mechanism.
2. **Rewritten LayerNorm**: Reduced LayerNorm execution time.
3. **Mixed-Precision Training**: Speedup in both training and inference.

For detailed results and benchmarks, see the [Experiments section](#experiments).

## Optimizations

### FlashAttention

FlashAttention is used to optimize the attention computation pipeline. It minimizes high-bandwidth memory (HBM) accesses by reorganizing the standard scaled-dot-product attention into a more efficient computation. This results in a significant speedup, especially when the sequence length is large.

For more information, check the [FlashAttention paper](https://arxiv.org/abs/2205.14135).

### Mixed-Precision Training

Mixed-precision training reduces memory consumption and increases throughput by performing most computations in FP16, with key operations like weight updates done in FP32. This optimization allows for faster training without sacrificing accuracy.

For details, refer to the [Mixed Precision Training paper](https://arxiv.org/abs/1710.03740).

### Rewritten LayerNorm

We replaced the custom LayerNorm implementation with PyTorch's `nn.LayerNorm`, which optimizes the execution by fusing operations into a single CUDA kernel, leading to faster computation times and reduced memory usage.

## Experiments

We conducted several experiments to evaluate the impact of the optimizations:

1. **Training Speed**: The combined optimizations result in up to **2.80× faster training**.
2. **Inference Speed**: Inference speedup reaches **2.95×**.
3. **Memory Efficiency**: Peak GPU memory consumption is reduced by **59%**.
4. **Accuracy**: The optimizations lead to a negligible increase of **0.61% in MSE**.

### Experiment Results

| Method                           | Speedup   |
| -------------------------------- | --------- |
| SynergyX (baseline)              | x1        |
| SynergyX + FlashAttention        | x1.29     |
| SynergyX + Rewritten LayerNorm   | x1.64     |
| SynergyX + Mixed Precision       | x1.86     |
| **SynergyX + All Optimizations** | **x2.80** |

