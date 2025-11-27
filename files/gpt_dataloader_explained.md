# GPT DataLoader Implementation: Complete Guide

## Overview

The `create_dataloader_v1()` function creates a PyTorch DataLoader that efficiently batches and loads training data for GPT models. It combines the `GPTDatasetV1` with PyTorch's `DataLoader` to enable batch processing, shuffling, and parallel data loading.

---

## Table of Contents

1. [PyTorch DataLoader Basics](#pytorch-dataloader-basics)
2. [The create_dataloader_v1 Function](#the-create_dataloader_v1-function)
3. [Parameter Deep Dive](#parameter-deep-dive)
4. [Batching Mechanism](#batching-mechanism)
5. [Complete Examples](#complete-examples)
6. [Training vs Validation DataLoaders](#training-vs-validation-dataloaders)
7. [Performance Optimization](#performance-optimization)

---

## PyTorch DataLoader Basics

### What is a DataLoader?

A `DataLoader` is a PyTorch utility that wraps a Dataset and provides:
- **Batching**: Groups multiple samples into batches
- **Shuffling**: Randomizes sample order (important for training)
- **Parallel Loading**: Uses multiple workers to load data faster
- **Automatic Batching**: Converts list of samples into batched tensors

### Why Use DataLoader?

**Without DataLoader (Manual):**
```python
# ❌ Manual batching - tedious and error-prone
for i in range(0, len(dataset), batch_size):
    batch_inputs = []
    batch_targets = []
    for j in range(batch_size):
        if i + j < len(dataset):
            inp, tgt = dataset[i + j]
            batch_inputs.append(inp)
            batch_targets.append(tgt)
    batch_inputs = torch.stack(batch_inputs)
    batch_targets = torch.stack(batch_targets)
    # Train on batch...
```

**With DataLoader (Automatic):**
```python
# ✅ Automatic batching - clean and efficient
for batch_inputs, batch_targets in dataloader:
    # Train on batch...
```

---

## The create_dataloader_v1 Function

### Complete Code

```python
from torch.utils.data import DataLoader
import tiktoken

def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    """
    Create a DataLoader for GPT training.
    
    Parameters:
    -----------
    txt : str
        Raw text data
    batch_size : int, default=4
        Number of samples per batch
    max_length : int, default=256
        Maximum sequence length
    stride : int, default=128
        Sliding window step size
    shuffle : bool, default=True
        Whether to shuffle data (use True for training)
    drop_last : bool, default=True
        Whether to drop incomplete final batch
    num_workers : int, default=0
        Number of parallel data loading workers
    
    Returns:
    --------
    DataLoader
        Configured PyTorch DataLoader
    """
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader
```

---

### Step-by-Step Breakdown

#### Step 1: Initialize Tokenizer

```python
tokenizer = tiktoken.get_encoding("gpt2")
```

**What it does:**
- Loads GPT-2 tokenizer (50,257 vocabulary size)
- Provides `encode()` and `decode()` methods
- Consistent with GPT-2 training

**Why inside the function?**
- ✅ Convenience: User doesn't need to pass tokenizer
- ✅ Consistency: Always uses GPT-2 tokenizer
- ⚠️ Less flexible: Can't use different tokenizers easily

**Alternative (more flexible):**
```python
def create_dataloader_v1(txt, tokenizer, batch_size=4, ...):
    # User passes tokenizer
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    ...
```

---

#### Step 2: Create Dataset

```python
dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
```

**What it does:**
- Tokenizes the text
- Creates sliding window samples
- Stores input-target pairs

**Example:**
```python
text = "Hello world! This is GPT."
dataset = GPTDatasetV1(text, tokenizer, max_length=4, stride=2)
print(len(dataset))  # e.g., 2 samples
```

**Memory consideration:**
- All samples created at initialization
- For large texts, can use lazy loading (see GPT Dataset guide)

---

#### Step 3: Create DataLoader

```python
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last,
    num_workers=num_workers
)
```

**What it does:**
- Wraps dataset with batching logic
- Configures shuffling and parallel loading
- Returns iterable that yields batches

---

## Parameter Deep Dive

### 1. batch_size

**Definition:** Number of samples grouped into each batch

**Impact on Training:**

```python
# batch_size = 2
Batch 1: [Sample 0, Sample 1]
Batch 2: [Sample 2, Sample 3]
Batch 3: [Sample 4, Sample 5]

# batch_size = 4
Batch 1: [Sample 0, Sample 1, Sample 2, Sample 3]
Batch 2: [Sample 4, Sample 5, Sample 6, Sample 7]
```

**Visual Example:**

```
Dataset with 10 samples, batch_size=3

Sample 0: [T0, T1, T2, T3]  ┐
Sample 1: [T4, T5, T6, T7]  ├─ Batch 0 (3 samples)
Sample 2: [T8, T9, T10, T11]┘

Sample 3: [T12, T13, T14, T15] ┐
Sample 4: [T16, T17, T18, T19] ├─ Batch 1 (3 samples)
Sample 5: [T20, T21, T22, T23] ┘

Sample 6: [T24, T25, T26, T27] ┐
Sample 7: [T28, T29, T30, T31] ├─ Batch 2 (3 samples)
Sample 8: [T32, T33, T34, T35] ┘

Sample 9: [T36, T37, T38, T39] ─── Batch 3 (1 sample) ← Incomplete!
```

**Choosing batch_size:**

| Batch Size | Memory Usage | Training Speed | Gradient Quality |
|------------|--------------|----------------|------------------|
| Small (1-4) | Low | Slow | Noisy (high variance) |
| Medium (8-32) | Medium | Moderate | Balanced |
| Large (64-128) | High | Fast | Smooth (low variance) |
| Very Large (256+) | Very High | Fastest | Very smooth (may hurt generalization) |

**Practical constraints:**

```python
# GPU Memory limits batch size
batch_size = 2   # Works on most laptops
batch_size = 8   # Needs decent GPU (8GB+)
batch_size = 32  # Needs high-end GPU (16GB+)
batch_size = 128 # Needs multiple GPUs or very large GPU

# Formula (rough estimate):
Memory (GB) ≈ batch_size × max_length × model_params / (1e9 × 4)
```

---

### 2. max_length

**Definition:** Maximum sequence length for each sample

**Impact:**

```python
# max_length = 4
Input:  [T0, T1, T2, T3]
Target: [T1, T2, T3, T4]

# max_length = 8
Input:  [T0, T1, T2, T3, T4, T5, T6, T7]
Target: [T1, T2, T3, T4, T5, T6, T7, T8]
```

**Trade-offs:**

| Aspect | Small max_length (64-128) | Large max_length (512-1024) |
|--------|---------------------------|------------------------------|
| Context | Limited | Full long-range context |
| Memory | Low | High |
| Speed | Fast | Slow |
| Samples | Many | Few |

**Example:**
```python
text = "A" * 10000
tokenizer = tiktoken.get_encoding("gpt2")

# Small max_length
dataset_small = GPTDatasetV1(text, tokenizer, max_length=128, stride=128)
print(len(dataset_small))  # ~78 samples

# Large max_length
dataset_large = GPTDatasetV1(text, tokenizer, max_length=512, stride=512)
print(len(dataset_large))  # ~19 samples
```

---

### 3. stride

**Definition:** Step size for sliding window

**Visual Impact:**

```
Text: [T0, T1, T2, T3, T4, T5, T6, T7, T8, T9]
max_length = 4

stride = 2 (50% overlap):
Sample 0: [T0, T1, T2, T3]
Sample 1:     [T2, T3, T4, T5]
Sample 2:         [T4, T5, T6, T7]
Sample 3:             [T6, T7, T8, T9]
Result: 4 samples with overlap

stride = 4 (no overlap):
Sample 0: [T0, T1, T2, T3]
Sample 1:                 [T4, T5, T6, T7]
Sample 2:                                 [T8, T9, ...]
Result: 2-3 samples, no overlap
```

**Common Patterns:**

```python
# 50% overlap (good balance)
stride = max_length // 2

# No overlap (efficient)
stride = max_length

# 75% overlap (maximum data augmentation)
stride = max_length // 4
```

---

### 4. shuffle

**Definition:** Whether to randomize sample order

**Impact on Training:**

**With shuffle=True (Training):**
```python
Epoch 1: [Sample 5, Sample 2, Sample 8, Sample 1, ...]
Epoch 2: [Sample 3, Sample 7, Sample 1, Sample 4, ...]
Epoch 3: [Sample 2, Sample 6, Sample 3, Sample 9, ...]

Each epoch sees samples in different order
Prevents overfitting to data order
```

**With shuffle=False (Validation):**
```python
Epoch 1: [Sample 0, Sample 1, Sample 2, Sample 3, ...]
Epoch 2: [Sample 0, Sample 1, Sample 2, Sample 3, ...]
Epoch 3: [Sample 0, Sample 1, Sample 2, Sample 3, ...]

Same order every time
Consistent evaluation
```

**Best Practices:**

```python
# Training: Always shuffle
train_loader = create_dataloader_v1(
    train_text,
    shuffle=True  # Randomize for better learning
)

# Validation: Never shuffle
val_loader = create_dataloader_v1(
    val_text,
    shuffle=False  # Consistent evaluation
)
```

**Why shuffle matters:**

```
Without shuffling:
- Model sees data in fixed order
- May learn spurious patterns based on order
- Overfits to sequence structure

With shuffling:
- Model sees varied combinations
- Learns more robust patterns
- Better generalization
```

---

### 5. drop_last

**Definition:** Whether to drop incomplete final batch

**Visual Example:**

```
Dataset: 10 samples, batch_size=3

Batch 0: [Sample 0, Sample 1, Sample 2] ✓ Full
Batch 1: [Sample 3, Sample 4, Sample 5] ✓ Full
Batch 2: [Sample 6, Sample 7, Sample 8] ✓ Full
Batch 3: [Sample 9]                      ⚠ Incomplete (size=1)

With drop_last=True:
  Batches: 3 (last one dropped)
  Total samples used: 9

With drop_last=False:
  Batches: 4 (all included)
  Total samples used: 10
```

**When to use each:**

```python
# Training: drop_last=True
# Reason: Consistent batch sizes for stable training
train_loader = create_dataloader_v1(
    train_text,
    batch_size=8,
    drop_last=True  # Drop incomplete batch
)

# Validation: drop_last=False
# Reason: Use all available data for evaluation
val_loader = create_dataloader_v1(
    val_text,
    batch_size=8,
    drop_last=False  # Keep all samples
)
```

**Impact on training stability:**

```python
# With drop_last=False
Batch 0: shape (8, 256)   # 8 samples
Batch 1: shape (8, 256)
Batch 2: shape (3, 256)   # Only 3 samples! ⚠

# Batch Normalization statistics unstable
# Gradient estimates vary widely
# Training can be noisy

# With drop_last=True
Batch 0: shape (8, 256)   # 8 samples
Batch 1: shape (8, 256)   # 8 samples
# Batch 2 dropped          # Stable throughout
```

---

### 6. num_workers

**Definition:** Number of parallel workers for data loading

**How it works:**

```
num_workers=0 (default):
Main Process: [Load Batch 1] → [Train] → [Load Batch 2] → [Train] → ...
              └─ GPU idle ────┘          └─ GPU idle ────┘

Total time = Loading time + Training time


num_workers=4:
Main Process:                [Train Batch 1] → [Train Batch 2] → ...
Worker 1:    [Load B1] [Load B5] ...
Worker 2:    [Load B2] [Load B6] ...
Worker 3:    [Load B3] [Load B7] ...
Worker 4:    [Load B4] [Load B8] ...
             └─ Overlap! GPU always busy ─┘

Total time ≈ Training time only (if loading is fast enough)
```

**Choosing num_workers:**

```python
import multiprocessing

# Safe default
num_workers = 0  # Single-threaded, simple debugging

# Good for CPU
num_workers = 2  # Basic parallelism

# Optimal (rough guideline)
num_workers = min(4, multiprocessing.cpu_count())

# Maximum (may be overkill)
num_workers = multiprocessing.cpu_count()
```

**Trade-offs:**

| num_workers | Pros | Cons |
|-------------|------|------|
| 0 | • Simple<br>• Easy debugging<br>• No overhead | • Slowest<br>• GPU waits for data |
| 2-4 | • Good speedup<br>• Manageable | • Some memory overhead<br>• Occasional issues on Windows |
| 8+ | • Maximum throughput | • High memory overhead<br>• Diminishing returns<br>• Complex debugging |

**Platform considerations:**

```python
# Windows: num_workers > 0 can be problematic
# Workaround: Keep it at 0 for simplicity
if os.name == 'nt':  # Windows
    num_workers = 0
else:  # Linux/Mac
    num_workers = 4
```

---

## Batching Mechanism

### How DataLoader Creates Batches

**Step-by-step process:**

```python
# Dataset with 6 samples
dataset[0] = (tensor([1, 2, 3]), tensor([2, 3, 4]))
dataset[1] = (tensor([5, 6, 7]), tensor([6, 7, 8]))
dataset[2] = (tensor([9, 10, 11]), tensor([10, 11, 12]))
dataset[3] = (tensor([13, 14, 15]), tensor([14, 15, 16]))
dataset[4] = (tensor([17, 18, 19]), tensor([18, 19, 20]))
dataset[5] = (tensor([21, 22, 23]), tensor([22, 23, 24]))

# Create DataLoader with batch_size=2
dataloader = DataLoader(dataset, batch_size=2)

# Iteration 1:
indices = [0, 1]  # First batch indices
samples = [dataset[0], dataset[1]]
# samples = [(tensor([1,2,3]), tensor([2,3,4])),
#            (tensor([5,6,7]), tensor([6,7,8]))]

# Stack inputs
inputs = torch.stack([samples[0][0], samples[1][0]])
# inputs = tensor([[1, 2, 3],
#                  [5, 6, 7]])  # Shape: (2, 3)

# Stack targets
targets = torch.stack([samples[0][1], samples[1][1]])
# targets = tensor([[2, 3, 4],
#                   [6, 7, 8]])  # Shape: (2, 3)

# Yield batch
yield inputs, targets
```

---

### Visual Batching Example

**Dataset:**
```
Sample 0: Input=[T0, T1, T2, T3], Target=[T1, T2, T3, T4]
Sample 1: Input=[T5, T6, T7, T8], Target=[T6, T7, T8, T9]
Sample 2: Input=[T10, T11, T12, T13], Target=[T11, T12, T13, T14]
Sample 3: Input=[T15, T16, T17, T18], Target=[T16, T17, T18, T19]
```

**Batching with batch_size=2:**

```
Batch 0:
Inputs:  [[T0, T1, T2, T3],     ← Sample 0
          [T5, T6, T7, T8]]     ← Sample 1
         Shape: (2, 4)

Targets: [[T1, T2, T3, T4],     ← Sample 0
          [T6, T7, T8, T9]]     ← Sample 1
         Shape: (2, 4)

Batch 1:
Inputs:  [[T10, T11, T12, T13],  ← Sample 2
          [T15, T16, T17, T18]]  ← Sample 3
         Shape: (2, 4)

Targets: [[T11, T12, T13, T14],  ← Sample 2
          [T16, T17, T18, T19]]  ← Sample 3
         Shape: (2, 4)
```

---

### Batch Shape Evolution

```python
# Single sample
input_ids, target_ids = dataset[0]
print(input_ids.shape)   # torch.Size([256]) - 1D

# Batched samples
for batch_inputs, batch_targets in dataloader:
    print(batch_inputs.shape)   # torch.Size([4, 256]) - 2D
    #                            ↑batch  ↑sequence
    print(batch_targets.shape)  # torch.Size([4, 256])
    break

# Through model
logits = model(batch_inputs)
print(logits.shape)  # torch.Size([4, 256, 50257])
#                     ↑batch ↑seq   ↑vocabulary
```

---

## Complete Examples

### Example 1: Basic Usage

```python
import torch

# Prepare text
text = """
The quick brown fox jumps over the lazy dog.
This is a sample text for training a GPT model.
"""

# Create DataLoader
dataloader = create_dataloader_v1(
    txt=text,
    batch_size=2,
    max_length=8,
    stride=4,
    shuffle=False,
    drop_last=False
)

print(f"Number of batches: {len(dataloader)}")

# Iterate through batches
for batch_idx, (inputs, targets) in enumerate(dataloader):
    print(f"\nBatch {batch_idx}:")
    print(f"  Inputs shape: {inputs.shape}")
    print(f"  Targets shape: {targets.shape}")
    print(f"  First input: {inputs[0, :5]}")  # First 5 tokens
    print(f"  First target: {targets[0, :5]}")
    
    # Verify shift relationship
    assert torch.equal(inputs[:, 1:], targets[:, :-1])
    print("  ✓ Shift verified")
```

**Output:**
```
Number of batches: 3

Batch 0:
  Inputs shape: torch.Size([2, 8])
  Targets shape: torch.Size([2, 8])
  First input: tensor([464, 2068, 7586, 21831, 18045])
  First target: tensor([2068, 7586, 21831, 18045, 625])
  ✓ Shift verified

Batch 1:
  Inputs shape: torch.Size([2, 8])
  Targets shape: torch.Size([2, 8])
  First input: tensor([625, 262, 16931, 3290, 13])
  First target: tensor([262, 16931, 3290, 13, 770])
  ✓ Shift verified

...
```

---

### Example 2: Training Loop

```python
import torch.nn as nn
import torch.optim as optim

# Load training text
with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
    train_text = f.read()

# Create DataLoader
train_loader = create_dataloader_v1(
    txt=train_text,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,      # Randomize for training
    drop_last=True,    # Consistent batch sizes
    num_workers=0
)

# Initialize model and optimizer
model = GPTModel(GPT_CONFIG_124M)
optimizer = optim.AdamW(model.parameters(), lr=0.0001)

# Training loop
model.train()
num_epochs = 5

for epoch in range(num_epochs):
    total_loss = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Forward pass
        logits = model(inputs)
        
        # Calculate loss
        loss = nn.functional.cross_entropy(
            logits.flatten(0, 1),
            targets.flatten()
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Log progress
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
```

---

### Example 3: Validation Evaluation

```python
def evaluate_model(model, dataloader):
    """Evaluate model on validation data."""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            # Forward pass
            logits = model(inputs)
            
            # Calculate loss
            loss = nn.functional.cross_entropy(
                logits.flatten(0, 1),
                targets.flatten()
            )
            
            batch_size = inputs.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    
    avg_loss = total_loss / total_samples
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    return avg_loss, perplexity.item()

# Create validation DataLoader
val_loader = create_dataloader_v1(
    txt=val_text,
    batch_size=4,
    max_length=256,
    stride=256,        # No overlap for validation
    shuffle=False,     # Consistent evaluation
    drop_last=False,   # Use all data
    num_workers=0
)

# Evaluate
val_loss, val_perplexity = evaluate_model(model, val_loader)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Perplexity: {val_perplexity:.2f}")
```

---

## Training vs Validation DataLoaders

### Key Differences

| Parameter | Training | Validation | Reason |
|-----------|----------|------------|--------|
| `shuffle` | `True` | `False` | Training needs randomization, validation needs consistency |
| `drop_last` | `True` | `False` | Training needs stable batches, validation needs all data |
| `stride` | `< max_length` | `= max_length` | Training benefits from overlap, validation avoids redundancy |

### Complete Setup Example

```python
# Load and split data
with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
    text_data = f.read()

train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_text = text_data[:split_idx]
val_text = text_data[split_idx:]

print(f"Training characters: {len(train_text)}")
print(f"Validation characters: {len(val_text)}")

# Training DataLoader
train_loader = create_dataloader_v1(
    txt=train_text,
    batch_size=4,
    max_length=256,
    stride=128,        # 50% overlap
    shuffle=True,      # Randomize
    drop_last=True,    # Consistent batches
    num_workers=0
)

# Validation DataLoader
val_loader = create_dataloader_v1(
    txt=val_text,
    batch_size=4,
    max_length=256,
    stride=256,        # No overlap
    shuffle=False,     # Deterministic
    drop_last=False,   # Use all data
    num_workers=0
)

print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
```

**Output:**
```
Training characters: 18532
Validation characters: 2059
Training batches: 15
Validation batches: 2
```

---

### Visual Comparison

**Training DataLoader (shuffle=True, drop_last=True, stride=128):**

```
Epoch 1:
[Batch 3] → [Batch 1] → [Batch 5] → [Batch 2] → ...
  Random order, maximizes learning

Samples with 50% overlap:
[───────Sample 0───────]
        [───────Sample 1───────]
                [───────Sample 2───────]
  More training data, better coverage

Last incomplete batch dropped:
Batch 0: ████████ (8 samples)
Batch 1: ████████ (8 samples)
Batch 2: ████████ (8 samples)
Batch 3: ███      (3 samples) ← DROPPED
  Stable training
```

**Validation DataLoader (shuffle=False, drop_last=False, stride=256):**

```
Every epoch:
[Batch 0] → [Batch 1] → [Batch 2] → ...
  Same order, consistent evaluation

Samples with no overlap:
[───────Sample 0───────][───────Sample 1───────][───────Sample 2───────]
  No redundancy, efficient evaluation

Last incomplete batch kept:
Batch 0: ████████ (8 samples)
Batch 1: ████████ (8 samples)
Batch 2: ███      (3 samples) ← KEPT
  Use all available data
```

---

## Performance Optimization

### Bottleneck Analysis

```python
import time

# Measure data loading time
start = time.time()
for batch_idx, (inputs, targets) in enumerate(train_loader):
    if batch_idx == 0:
        first_batch_time = time.time() - start
    pass
total_time = time.time() - start

print(f"First batch: {first_batch_time:.3f}s")
print(f"Total time: {total_time:.3f}s")
print(f"Avg per batch: {total_time/len(train_loader):.3f}s")
```

### Optimization Strategies

#### 1. Increase num_workers

```python
# Before (slow)
dataloader = create_dataloader_v1(
    txt=text,
    batch_size=8,
    num_workers=0  # Single-threaded
)
# Time per epoch: 45s

# After (faster)
dataloader = create_dataloader_v1(
    txt=text,
    batch_size=8,
    num_workers=4  # Parallel loading
)
# Time per epoch: 32s (1.4x speedup)
```

#### 2. Adjust stride for validation

```python
# Before (redundant)
val_loader = create_dataloader_v1(
    txt=val_text,
    stride=128  # 50% overlap
)
# Batches: 20, many redundant evaluations

# After (efficient)
val_loader = create_dataloader_v1(
    txt=val_text,
    stride=256  # No overlap
)
# Batches: 10, faster validation
```

#### 3. Use pin_memory for GPU training

```python
# Modified DataLoader creation
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last,
    num_workers=num_workers,
    pin_memory=True  # Faster GPU transfer
)
```

#### 4. Prefetch batches

```python
# Use persistent_workers for faster epoch transitions
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last,
    num_workers=4,
    persistent_workers=True  # Keep workers alive
)
```

---

## Common Issues and Solutions

### Issue 1: Out of Memory

```python
# ❌ Problem
dataloader = create_dataloader_v1(text, batch_size=64, max_length=1024)
# RuntimeError: CUDA out of memory

# ✅ Solution 1: Reduce batch_size
dataloader = create_dataloader_v1(text, batch_size=8, max_length=1024)

# ✅ Solution 2: Reduce max_length
dataloader = create_dataloader_v1(text, batch_size=64, max_length=256)

# ✅ Solution 3: Gradient accumulation
for batch_idx, (inputs, targets) in enumerate(dataloader):
    logits = model(inputs)
    loss = calculate_loss(logits, targets) / accumulation_steps
    loss.backward()
    
    if (batch_idx + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Issue 2: Too Few Batches

```python
# Problem: Only 2 batches from large dataset
dataloader = create_dataloader_v1(
    text,
    max_length=1024,
    stride=5000  # Stride too large!
)

# Solution: Reduce stride
dataloader = create_dataloader_v1(
    text,
    max_length=1024,
    stride=512  # More reasonable
)
```

### Issue 3: Slow Data Loading

```python
# ❌ Slow
dataloader = create_dataloader_v1(text, num_workers=0)

# ✅ Faster
dataloader = create_dataloader_v1(text, num_workers=4)

# ⚠️ Windows users
if os.name == 'nt':
    # Keep num_workers=0 to avoid multiprocessing issues
    dataloader = create_dataloader_v1(text, num_workers=0)
```

### Issue 4: Inconsistent Validation Results

```python
# ❌ Problem: Results vary between runs
val_loader = create_dataloader_v1(val_text, shuffle=True)

# ✅ Solution: Disable shuffling
val_loader = create_dataloader_v1(val_text, shuffle=False)
```

---

## Summary

### Key Concepts

1. **DataLoader**: Automates batching, shuffling, and parallel loading
2. **Batching**: Groups samples into tensors for efficient processing
3. **Shuffling**: Randomizes order for training, disabled for validation
4. **drop_last**: Ensures consistent batch sizes for training
5. **num_workers**: Parallelizes data loading for speed

### Quick Reference

```python
# Training DataLoader
train_loader = create_dataloader_v1(
    txt=train_text,
    batch_size=8,          # GPU memory dependent
    max_length=256,        # Model context window
    stride=128,            # 50% overlap
    shuffle=True,          # Randomize order
    drop_last=True,        # Consistent batches
    num_workers=2          # Parallel loading
)

# Validation DataLoader
val_loader = create_dataloader_v1(
    txt=val_text,
    batch_size=8,
    max_length=256,
    stride=256,            # No overlap
    shuffle=False,         # Deterministic
    drop_last=False,       # Use all data
    num_workers=2
)

# Usage
for inputs, targets in train_loader:
    # inputs: (batch_size, max_length)
    # targets: (batch_size, max_length)
    logits = model(inputs)
    loss = calculate_loss(logits, targets)
    # ... training step
```

### Best Practices

✅ **DO:**
- Use `shuffle=True` for training
- Use `shuffle=False` for validation
- Use `drop_last=True` for training
- Use `drop_last=False` for validation
- Set `stride < max_length` for training (overlap)
- Set `stride = max_length` for validation (efficiency)
- Use `num_workers > 0` on Linux/Mac for speed

❌ **DON'T:**
- Shuffle validation data
- Drop last batch in validation
- Use very large batch_size (GPU OOM)
- Use stride > max_length (wastes data)
- Use too many workers (diminishing returns)

The DataLoader is the bridge between your dataset and model, enabling efficient, batched training of GPT models!
