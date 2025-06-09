# SynergyX Optimization: Efficient Drug Synergy Prediction

This repository contains the implementation of **SynergyX**, a multi-modality mutual attention network designed for interpretable drug synergy prediction. We have optimized the SynergyX framework to enhance computational efficiency while maintaining high prediction accuracy for large-scale drug synergy prediction. The key optimizations include **FlashAttention**, **mixed-precision training**, and **rewritten LayerNorm**. These improvements have led to remarkable performance gains, including:

- **Training Speed**: **2.80× faster**
- **Inference Speed**: **2.95× faster**
- **Memory Efficiency**: **59% reduction**
- **Accuracy**: **0.61% increase in MSE**

These optimizations significantly enhance the training and inference speed while reducing memory consumption, all with a minimal impact on model accuracy.

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
mkdir experiment
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

### Experiment Results

We conducted several experiments to evaluate the impact of the optimizations. The results are summarized below:

### 1. **Training Speed**

The optimizations resulted in **up to 2.80× faster training**.

| Method                           | Time (min/epoch) | Speedup   |
| -------------------------------- | ---------------- | --------- |
| SynergyX (baseline)              | 2246.47 (338)    | x1        |
| SynergyX + Flash Attention       | 1654.42 (322)    | x1.29     |
| SynergyX + Rewritten LayerNorm   | 986.40 (244)     | x1.64     |
| SynergyX + Mixed Precision       | 1022.14 (286)    | x1.86     |
| **SynergyX + All Optimizations** | **877.23 (370)** | **x2.80** |

### 2. **Inference Speed**

The optimizations led to a **2.95× speedup** in inference time.

| Configuration | Batch Size | Time (s) | Speedup   |
| ------------- | ---------- | -------- | --------- |
| SynergyX      | 32         | 6.7847   | 1x        |
| Our Method    | 32         | 6.6702   | 1.02x     |
| SynergyX      | 64         | 5.3384   | 1x        |
| Our Method    | 64         | 4.4091   | 1.21x     |
| SynergyX      | 128        | 4.9827   | 1x        |
| Our Method    | 128        | 2.1349   | 2.33x     |
| SynergyX      | 256        | 4.8026   | 1x        |
| Our Method    | 256        | 1.6255   | **2.95x** |

### 3. **Memory Efficiency**

Peak GPU memory consumption was reduced by **59%** with our method.

| Method         | Batch Size | Memory Used (MiB) | Memory Used (%) |
| -------------- | ---------- | ----------------- | --------------- |
| SynergyX       | 512        | 28,329            | 100%            |
| **Our Method** | 512        | **11,429**        | **40.34%**      |

### 4. **Accuracy**

The optimizations resulted in a negligible **0.61% increase in MSE**.

| Method                    | MSE (Test)  | Time (epochs)            | Speedup   | Precision Loss |
| ------------------------- | ----------- | ------------------------ | --------- | -------------- |
| SynergyX                  | **79.7689** | 2246.47 min (338 epochs) | x1        | 0%             |
| Our Method                | 81.3122     | 622.98 min (255 epochs)  | x2.71     | -1.93%         |
| Our Method + Weight Decay | **80.2572** | 877.23 min (370 epochs)  | **x2.80** | **-0.61%**     |

------

### Conclusion

- **Training Speed**: **2.80× faster**
- **Inference Speed**: **2.95× faster**
- **Memory Efficiency**: **59% reduction**
- **Accuracy**: **0.61% increase in MSE**

These results demonstrate that the proposed optimizations significantly improve both performance and memory efficiency with minimal impact on accuracy.
