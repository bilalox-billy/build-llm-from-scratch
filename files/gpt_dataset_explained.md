# GPT Dataset Implementation: Complete Guide

## Overview

The `GPTDatasetV1` class is a custom PyTorch Dataset that prepares text data for training GPT models. It implements a **sliding window approach** to create overlapping training sequences, where each input sequence is paired with its corresponding target sequence (shifted by one token for next-token prediction).

---

## Table of Contents

1. [PyTorch Dataset Basics](#pytorch-dataset-basics)
2. [The GPTDatasetV1 Class](#the-gptdatasetv1-class)
3. [Sliding Window Mechanism](#sliding-window-mechanism)
4. [Input-Target Relationship](#input-target-relationship)
5. [Complete Examples](#complete-examples)
6. [Parameter Impact Analysis](#parameter-impact-analysis)
7. [Memory Considerations](#memory-considerations)

---

## PyTorch Dataset Basics

### What is a PyTorch Dataset?

A PyTorch `Dataset` is an abstract class that represents a dataset. Any custom dataset must inherit from `torch.utils.data.Dataset` and implement three methods:

```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self):
        # Initialize data structures
        pass
    
    def __len__(self):
        # Return the total number of samples
        return num_samples
    
    def __getitem__(self, idx):
        # Return the sample at index idx
        return sample
```

**Why use Dataset?**
- Standardized interface for data loading
- Works seamlessly with DataLoader for batching
- Enables efficient memory usage (loads data on-demand)
- Supports multiprocessing for faster loading

---

## The GPTDatasetV1 Class

### Complete Code

```python
from torch.utils.data import Dataset
import torch

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
```

---

### Method-by-Method Breakdown

#### `__init__` Method

**Purpose:** Initialize the dataset by tokenizing text and creating training sequences.

**Parameters:**
- `txt` (str): Raw text to be processed
- `tokenizer`: tiktoken tokenizer for GPT-2
- `max_length` (int): Maximum sequence length (e.g., 256 tokens)
- `stride` (int): Step size for sliding window (e.g., 128 tokens)

**Step-by-Step Process:**

```python
def __init__(self, txt, tokenizer, max_length, stride):
    # Step 1: Initialize empty lists
    self.input_ids = []    # Will store input sequences
    self.target_ids = []   # Will store target sequences (shifted by 1)
```

**Why empty lists?**
- We'll populate them dynamically as we slide through the text
- Allows flexible dataset size based on text length

```python
    # Step 2: Tokenize entire text
    token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
```

**Example:**
```python
txt = "The quick brown fox jumps over the lazy dog"
token_ids = [464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290]
#            "The" "quick" "brown" "fox" "jumps" "over" "the" "lazy" "dog"
```

```python
    # Step 3: Create sliding windows
    for i in range(0, len(token_ids) - max_length, stride):
        input_chunk = token_ids[i:i + max_length]
        target_chunk = token_ids[i + 1: i + max_length + 1]
        self.input_ids.append(torch.tensor(input_chunk))
        self.target_ids.append(torch.tensor(target_chunk))
```

**Loop Analysis:**

```python
range(0, len(token_ids) - max_length, stride)
       ↓        ↓                      ↓
     start    stop                  step

# Example: len(token_ids)=100, max_length=10, stride=5
# range(0, 90, 5) → [0, 5, 10, 15, 20, ..., 85]
# Creates 18 overlapping windows
```

**Why `len(token_ids) - max_length`?**

Ensures we have enough tokens for both input AND target:
```
Total tokens: 100
max_length: 10

Last valid start position: 90
- Input needs: tokens[90:100] = 10 tokens ✓
- Target needs: tokens[91:101] = 10 tokens ✓

If we started at 91:
- Input: tokens[91:101] = 10 tokens ✓
- Target: tokens[92:102] = ERROR! Index 101 doesn't exist ✗
```

---

#### `__len__` Method

**Purpose:** Return the total number of training samples.

```python
def __len__(self):
    return len(self.input_ids)
```

**Example:**
```python
dataset = GPTDatasetV1(text, tokenizer, max_length=10, stride=5)
print(len(dataset))  # Output: 18 (number of windows created)
```

**Why needed?**
- DataLoader uses this to know how many batches to create
- Enables iteration: `for i in range(len(dataset))`

---

#### `__getitem__` Method

**Purpose:** Retrieve a specific training sample by index.

```python
def __getitem__(self, idx):
    return self.input_ids[idx], self.target_ids[idx]
```

**Example:**
```python
dataset = GPTDatasetV1(text, tokenizer, max_length=10, stride=5)

# Get first sample
input_seq, target_seq = dataset[0]
print(input_seq)   # tensor([464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13])
print(target_seq)  # tensor([2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13, 383])
```

**Why needed?**
- Enables indexing: `dataset[5]`
- DataLoader calls this internally when creating batches
- Supports random sampling when `shuffle=True`

---

## Sliding Window Mechanism

### Concept

A **sliding window** moves across the text with a fixed step size (stride), extracting sequences of a fixed length (max_length).

### Visual Representation

**Example Text:**
```
"The quick brown fox jumps over the lazy dog and runs away"
Token IDs: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
```

**Parameters:**
- `max_length = 5`
- `stride = 3`

**Sliding Window Process:**

```
Position 0 (i=0):
[1, 2, 3, 4, 5] 6, 7, 8, 9, 10, 11, 12, 13
└─────────────┘
  Window 1

Position 1 (i=3):
1, 2, 3, [4, 5, 6, 7, 8] 9, 10, 11, 12, 13
         └─────────────┘
           Window 2

Position 2 (i=6):
1, 2, 3, 4, 5, 6, [7, 8, 9, 10, 11] 12, 13
                  └──────────────┘
                     Window 3

Position 3 (i=9):  ← STOP HERE
1, 2, 3, 4, 5, 6, 7, 8, 9, [10, 11, 12, 13] ❌
                           └──────────────┘
                           Only 4 tokens! Need 5
```

**Result:** 3 windows created

---

### Stride Impact

**Small Stride (More Overlap):**

```
stride = 2, max_length = 5

Window 1: [1, 2, 3, 4, 5]
Window 2:    [3, 4, 5, 6, 7]
Window 3:       [5, 6, 7, 8, 9]
Window 4:          [7, 8, 9, 10, 11]

Overlap: 3 tokens between consecutive windows
More training samples (4 windows)
```

**Large Stride (Less Overlap):**

```
stride = 5, max_length = 5

Window 1: [1, 2, 3, 4, 5]
Window 2:                [6, 7, 8, 9, 10]
Window 3:                               [11, 12, 13, 14, 15]

Overlap: 0 tokens (no overlap)
Fewer training samples (3 windows)
```

**Stride = max_length (No Overlap):**

```
stride = 5, max_length = 5

[1, 2, 3, 4, 5] | [6, 7, 8, 9, 10] | [11, 12, 13, 14, 15]
└──────────────┘   └───────────────┘   └────────────────┘
   Window 1            Window 2             Window 3

Perfect segmentation, no token reuse
```

---

### Complete Visual Example

**Input Text:**
```
"Every effort moves you closer to your goals"
```

**Tokenization:**
```
Token IDs: [6109, 3626, 6100, 345, 5699, 284, 534, 4661]
           "Every" "effort" "moves" "you" "closer" "to" "your" "goals"
```

**Parameters:** `max_length = 4`, `stride = 2`

**Window Creation:**

```
Iteration 0 (i=0):
Token IDs:  [6109, 3626, 6100, 345, 5699, 284, 534, 4661]
             └────────────────────┘
Input:       [6109, 3626, 6100, 345]        "Every effort moves you"
Target:           [3626, 6100, 345, 5699]   "effort moves you closer"
                   ↑
            Shifted by 1 position

Iteration 1 (i=2):
Token IDs:  [6109, 3626, 6100, 345, 5699, 284, 534, 4661]
                   └────────────────────┘
Input:             [6100, 345, 5699, 284]   "moves you closer to"
Target:                 [345, 5699, 284, 534] "you closer to your"

Iteration 2 (i=4):
Token IDs:  [6109, 3626, 6100, 345, 5699, 284, 534, 4661]
                               └────────────────────┘
Input:                         [5699, 284, 534, 4661] "closer to your goals"
Target:                             [284, 534, 4661, ???] ERROR!
                                                      ↑
                                              Need token at index 8 (doesn't exist)

So iteration 2 is SKIPPED (i=4 exceeds len(token_ids) - max_length = 8 - 4 = 4)
```

**Final Dataset:**
```python
self.input_ids = [
    tensor([6109, 3626, 6100, 345]),    # Window 0
    tensor([6100, 345, 5699, 284])      # Window 1
]

self.target_ids = [
    tensor([3626, 6100, 345, 5699]),    # Window 0 targets
    tensor([345, 5699, 284, 534])       # Window 1 targets
]

len(dataset) = 2
```

---

## Input-Target Relationship

### Why Shift by One?

GPT models are trained for **next-token prediction**. Given a sequence of tokens, the model must predict the next token at each position.

**Training Objective:**

```
Given:   "Every effort moves"
Predict: "effort moves you"

Position by position:
Input:  "Every"  → Target: "effort"
Input:  "effort" → Target: "moves"
Input:  "moves"  → Target: "you"
```

### Code Explanation

```python
input_chunk = token_ids[i:i + max_length]
target_chunk = token_ids[i + 1: i + max_length + 1]
```

**Visual Breakdown:**

```
token_ids = [T0, T1, T2, T3, T4, T5, T6, T7, T8, T9]
i = 2, max_length = 4

input_chunk = token_ids[2:6]     → [T2, T3, T4, T5]
target_chunk = token_ids[3:7]    → [T3, T4, T5, T6]

Alignment:
Input:  [T2, T3, T4, T5]
Target:     [T3, T4, T5, T6]
         ↑   ↑   ↑   ↑
      Predict each next token
```

**Position-by-Position:**

```
Model sees T2, predicts T3 ✓
Model sees T2→T3, predicts T4 ✓
Model sees T2→T3→T4, predicts T5 ✓
Model sees T2→T3→T4→T5, predicts T6 ✓
```

---

### Detailed Example

```python
# Sample text
text = "I love machine learning and AI"
token_ids = [40, 1842, 4572, 4673, 290, 9552]
#            "I" "love" "machine" "learning" "and" "AI"

# Parameters
max_length = 3
stride = 1
```

**Window Creation:**

```
Window 0 (i=0):
Input:  [40, 1842, 4572]           "I love machine"
Target:     [1842, 4572, 4673]     "love machine learning"

Token-by-token predictions:
Position 0: Given "I"              → Predict "love"
Position 1: Given "I love"         → Predict "machine"
Position 2: Given "I love machine" → Predict "learning"

Window 1 (i=1):
Input:  [1842, 4572, 4673]         "love machine learning"
Target:       [4572, 4673, 290]    "machine learning and"

Token-by-token predictions:
Position 0: Given "love"                     → Predict "machine"
Position 1: Given "love machine"             → Predict "learning"
Position 2: Given "love machine learning"    → Predict "and"

Window 2 (i=2):
Input:  [4572, 4673, 290]          "machine learning and"
Target:       [4673, 290, 9552]    "learning and AI"

Token-by-token predictions:
Position 0: Given "machine"                  → Predict "learning"
Position 1: Given "machine learning"         → Predict "and"
Position 2: Given "machine learning and"     → Predict "AI"
```

---

## Complete Examples

### Example 1: Small Dataset

```python
import tiktoken
import torch

# Setup
text = "Hello world! This is a test."
tokenizer = tiktoken.get_encoding("gpt2")

# Create dataset
dataset = GPTDatasetV1(
    txt=text,
    tokenizer=tokenizer,
    max_length=4,
    stride=2
)

# Inspect
print(f"Dataset size: {len(dataset)}")
print(f"Number of samples: {len(dataset)}")

# Examine first sample
input_ids, target_ids = dataset[0]
print(f"\nSample 0:")
print(f"Input shape: {input_ids.shape}")
print(f"Input IDs: {input_ids}")
print(f"Input text: {tokenizer.decode(input_ids.tolist())}")
print(f"\nTarget IDs: {target_ids}")
print(f"Target text: {tokenizer.decode(target_ids.tolist())}")
```

**Output:**
```
Dataset size: 2
Number of samples: 2

Sample 0:
Input shape: torch.Size([4])
Input IDs: tensor([15496, 995, 0, 770])
Input text: Hello world! This

Target IDs: tensor([995, 0, 770, 318])
Target text:  world! This is
```

---

### Example 2: Real Training Scenario

```python
# Load larger text
with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(f"Total characters: {len(text)}")

# Create dataset
tokenizer = tiktoken.get_encoding("gpt2")
dataset = GPTDatasetV1(
    txt=text,
    tokenizer=tokenizer,
    max_length=256,    # GPT-2 context window (reduced from 1024)
    stride=128         # 50% overlap
)

print(f"Total samples: {len(dataset)}")

# Analyze first few samples
for i in range(3):
    input_ids, target_ids = dataset[i]
    print(f"\nSample {i}:")
    print(f"Input length: {len(input_ids)}")
    print(f"Target length: {len(target_ids)}")
    print(f"First 10 input tokens: {input_ids[:10]}")
    print(f"First 10 target tokens: {target_ids[:10]}")
    
    # Verify shift relationship
    assert torch.equal(input_ids[1:], target_ids[:-1]), "Mismatch!"
    print("✓ Shift verification passed")
```

---

### Example 3: Stride Impact Comparison

```python
text = "A" * 1000  # Simple text for demonstration
tokenizer = tiktoken.get_encoding("gpt2")
max_length = 100

# Different stride values
strides = [50, 100, 150, 200]

for stride in strides:
    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)
    overlap = max(0, max_length - stride)
    print(f"Stride: {stride:3d} | Samples: {len(dataset):3d} | Overlap: {overlap} tokens")
```

**Output:**
```
Stride:  50 | Samples:  19 | Overlap: 50 tokens (50% overlap)
Stride: 100 | Samples:  10 | Overlap:  0 tokens (no overlap)
Stride: 150 | Samples:   7 | Overlap:  0 tokens (gaps of 50 tokens)
Stride: 200 | Samples:   5 | Overlap:  0 tokens (gaps of 100 tokens)
```

---

## Parameter Impact Analysis

### max_length Impact

**Purpose:** Controls sequence length model sees during training

| max_length | Pros | Cons |
|------------|------|------|
| Small (64-128) | • Faster training<br>• Less memory<br>• More samples | • Limited context<br>• May miss long-range dependencies |
| Medium (256-512) | • Balanced<br>• Good for most tasks | • Moderate memory use |
| Large (1024-2048) | • Full context<br>• Captures long dependencies | • High memory use<br>• Slower training<br>• Fewer samples |

**Example:**
```
Text: 10,000 tokens
max_length: 256, stride: 256

Number of samples = (10,000 - 256) / 256 = 38 samples
Each sample uses 256 tokens
```

---

### stride Impact

**Purpose:** Controls overlap between consecutive windows

| stride | Effect | Use Case |
|--------|--------|----------|
| stride < max_length | Overlapping windows | • More training data<br>• Better learning<br>• Data augmentation |
| stride = max_length | No overlap | • Efficient use of data<br>• No redundancy<br>• Standard training |
| stride > max_length | Gaps (tokens skipped) | • Rare<br>• Wastes data<br>• Generally avoided |

**Overlap Calculation:**
```python
overlap = max(0, max_length - stride)

# Examples:
max_length=256, stride=128 → overlap=128 (50% overlap)
max_length=256, stride=256 → overlap=0   (no overlap)
max_length=256, stride=512 → overlap=0   (gap of 256 tokens)
```

---

### Memory Footprint

**Storage calculation:**

```python
# Single sample memory
memory_per_sample = max_length * 2 * 8 bytes  # input + target, int64
                  = max_length * 16 bytes

# Total dataset memory
num_samples = (len(token_ids) - max_length) // stride
total_memory = num_samples * memory_per_sample

# Example: 10,000 tokens, max_length=256, stride=128
num_samples = (10,000 - 256) / 128 = 76
total_memory = 76 * 256 * 16 = 311,296 bytes ≈ 304 KB
```

---

## Memory Considerations

### Dataset Storage Strategy

**Current Implementation (In-Memory):**

```python
# All samples stored in memory
self.input_ids = []
self.target_ids = []

for i in range(...):
    self.input_ids.append(torch.tensor(input_chunk))
    self.target_ids.append(torch.tensor(target_chunk))
```

**Pros:**
- ✅ Fast access during training
- ✅ Simple implementation
- ✅ Good for small-medium datasets

**Cons:**
- ❌ High memory usage for large texts
- ❌ All samples loaded at initialization

---

### Alternative: On-the-Fly Generation

**For very large datasets:**

```python
class GPTDatasetV2(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        self.max_length = max_length
        self.stride = stride
        # Don't pre-generate samples!
    
    def __len__(self):
        return (len(self.token_ids) - self.max_length) // self.stride + 1
    
    def __getitem__(self, idx):
        # Generate sample on-the-fly
        start_idx = idx * self.stride
        input_chunk = self.token_ids[start_idx:start_idx + self.max_length]
        target_chunk = self.token_ids[start_idx + 1:start_idx + self.max_length + 1]
        return torch.tensor(input_chunk), torch.tensor(target_chunk)
```

**Benefits:**
- Lower memory (only stores token_ids once)
- Slower per-sample access (creates tensors on demand)

---

## Common Patterns and Best Practices

### Pattern 1: Standard Training Setup

```python
# 50% overlap for good data utilization
dataset = GPTDatasetV1(
    txt=train_text,
    tokenizer=tokenizer,
    max_length=256,
    stride=128  # 50% overlap
)
```

### Pattern 2: No Overlap (Efficient)

```python
# No overlap, maximum efficiency
dataset = GPTDatasetV1(
    txt=train_text,
    tokenizer=tokenizer,
    max_length=256,
    stride=256  # stride = max_length
)
```

### Pattern 3: Maximum Augmentation

```python
# Small stride for maximum training samples
dataset = GPTDatasetV1(
    txt=train_text,
    tokenizer=tokenizer,
    max_length=256,
    stride=1  # Every possible position (very slow!)
)
```

---

## Troubleshooting

### Issue 1: Empty Dataset

```python
dataset = GPTDatasetV1(text, tokenizer, max_length=1000, stride=500)
print(len(dataset))  # Output: 0

# Problem: Text is too short!
# Solution: Reduce max_length or use longer text
```

### Issue 2: Too Few Samples

```python
# Only 2 samples from 1000 tokens?
dataset = GPTDatasetV1(text, tokenizer, max_length=256, stride=512)

# Problem: stride > max_length creates gaps
# Solution: Use stride ≤ max_length
```

### Issue 3: Memory Error

```python
# MemoryError: Dataset too large
dataset = GPTDatasetV1(very_large_text, tokenizer, max_length=1024, stride=1)

# Problem: Too many samples (stride=1 creates millions)
# Solution: Increase stride or use on-the-fly generation
```

---

## Summary

### Key Concepts

1. **Sliding Window**: Creates overlapping training sequences from continuous text
2. **Input-Target Shift**: Targets are inputs shifted by 1 for next-token prediction
3. **Stride Control**: Balances data efficiency vs. training samples
4. **Memory Trade-off**: Pre-generation (fast) vs. on-the-fly (memory-efficient)

### Quick Reference

```python
# Create dataset
dataset = GPTDatasetV1(
    txt=text,                # Raw text string
    tokenizer=tokenizer,     # tiktoken encoder
    max_length=256,          # Sequence length
    stride=128              # Window step size
)

# Access samples
input_ids, target_ids = dataset[0]
print(f"Dataset size: {len(dataset)}")

# Verify shift relationship
assert torch.equal(input_ids[1:], target_ids[:-1])
```

### Formula Summary

```python
# Number of samples
num_samples = (total_tokens - max_length) // stride + 1

# Overlap between windows
overlap = max(0, max_length - stride)

# Memory per sample (bytes)
memory = max_length * 2 * 8  # input + target, int64
```

This dataset class is the foundation for efficient GPT training, enabling the model to learn next-token prediction from continuous text data!
