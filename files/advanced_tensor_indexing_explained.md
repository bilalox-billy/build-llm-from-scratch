# Advanced Tensor Indexing in PyTorch: Complete Guide

## Overview

This guide explains advanced tensor indexing techniques in PyTorch, focusing on multi-dimensional indexing with lists. These techniques are essential for extracting specific elements from complex tensors in deep learning applications.

---

## Table of Contents

1. [Basic Indexing Review](#basic-indexing-review)
2. [The `probas[0, [0,1,2], :]` Pattern](#the-probas0-012--pattern)
3. [Fancy Indexing with Lists](#fancy-indexing-with-lists)
4. [Practical Examples](#practical-examples)
5. [Target Probability Extraction](#target-probability-extraction)
6. [Common Patterns](#common-patterns)
7. [Performance Considerations](#performance-considerations)

---

## Basic Indexing Review

### Single Element Access

```python
# Create a 3D tensor
tensor = torch.randn(2, 3, 4)  # (batch, sequence, features)

# Access single element
element = tensor[0, 1, 2]  # Batch 0, Position 1, Feature 2
print(element.shape)  # torch.Size([]) - scalar
```

### Slice Access

```python
# Get all features for first batch, second position
features = tensor[0, 1, :]
print(features.shape)  # torch.Size([4])

# Get all positions for first batch
positions = tensor[0, :, :]
print(positions.shape)  # torch.Size([3, 4])

# Simplified notation
positions = tensor[0]  # Same as tensor[0, :, :]
```

---

## The `probas[0, [0,1,2], :]` Pattern

### Context: Model Probabilities

In GPT models, after a forward pass, we get:

```python
# Model output shape
logits.shape    # (batch_size, num_tokens, vocab_size)
                # (2, 3, 50257) in our example

# After softmax
probas = torch.softmax(logits, dim=-1)
probas.shape    # Still (2, 3, 50257)
```

**What this represents:**
- **Dimension 0**: Which training example (batch)
- **Dimension 1**: Which token position
- **Dimension 2**: Probability for each vocabulary token

### Breaking Down `probas[0, [0,1,2], :]`

```python
probas[0, [0,1,2], :]
       ↓    ↓       ↓
       │    │       └─ All vocabulary probabilities (50,257 values)
       │    └───────── Tokens at positions 0, 1, and 2
       └────────────── Batch 0 (first training example)
```

**Step-by-step extraction:**

1. **`probas[0]`** → Select first batch
   - Shape becomes: `(3, 50257)` instead of `(2, 3, 50257)`

2. **`[0,1,2]`** → Select specific token positions
   - This is a **list**, not a slice!
   - Tells PyTorch: "Give me rows 0, 1, and 2"

3. **`[:]`** → Keep all vocabulary dimensions
   - Take all 50,257 probability values

**Result:** `(3, 50257)` tensor with probability distributions for tokens 0, 1, and 2 from batch 0

---

## Visual Representation

### Original Tensor Structure

```
probas shape: (2, 3, 50257)

                    Batch 0                              Batch 1
    ┌──────────────────────────────────┐   ┌──────────────────────────────┐
    │ Token 0: [p₀, p₁, ..., p₅₀₂₅₆]  │   │ Token 0: [...]               │
    │ Token 1: [p₀, p₁, ..., p₅₀₂₅₆]  │   │ Token 1: [...]               │
    │ Token 2: [p₀, p₁, ..., p₅₀₂₅₆]  │   │ Token 2: [...]               │
    └──────────────────────────────────┘   └──────────────────────────────┘
```

### After `probas[0, [0,1,2], :]`

```
Result shape: (3, 50257)

    ┌──────────────────────────────────┐
    │ Token 0: [p₀, p₁, ..., p₅₀₂₅₆]  │  ← From batch 0, position 0
    │ Token 1: [p₀, p₁, ..., p₅₀₂₅₆]  │  ← From batch 0, position 1
    │ Token 2: [p₀, p₁, ..., p₅₀₂₅₆]  │  ← From batch 0, position 2
    └──────────────────────────────────┘
```

### Detailed Probability Distribution

```
Each row is a probability distribution over 50,257 tokens:

Token Position 0:
[0.00001, 0.00003, 0.00002, ..., 0.00098, ..., 0.00001]
   ↑         ↑         ↑              ↑              ↑
  ID 0      ID 1      ID 2        ID 3626       ID 50256
  "!"       "."      "#"          "effort"    <|endoftext|>

Sum of all probabilities = 1.0
```

---

## Fancy Indexing with Lists

### Why Use Lists Instead of Slices?

**Slices** (e.g., `0:3`) select contiguous ranges:
```python
probas[0, 0:3, :]  # Tokens 0, 1, 2 (contiguous)
```

**Lists** (e.g., `[0,1,2]`) can select **non-contiguous** or **reordered** indices:
```python
probas[0, [0, 2], :]     # Skip token 1
probas[0, [2, 0, 1], :]  # Reorder tokens
probas[0, [1, 1, 1], :]  # Repeat token 1 three times
```

### Examples of List Indexing

#### Example 1: Non-Contiguous Selection

```python
# Select only first and last token (skip middle)
selected = probas[0, [0, 2], :]
print(selected.shape)  # torch.Size([2, 50257])

# Result contains:
# Row 0: Probabilities for token at position 0
# Row 1: Probabilities for token at position 2
# (Position 1 is skipped)
```

#### Example 2: Reverse Order

```python
# Get tokens in reverse order
reversed_tokens = probas[0, [2, 1, 0], :]
print(reversed_tokens.shape)  # torch.Size([3, 50257])

# Result:
# Row 0: Originally position 2
# Row 1: Originally position 1
# Row 2: Originally position 0
```

#### Example 3: Single Token

```python
# Select only middle token
middle = probas[0, [1], :]
print(middle.shape)  # torch.Size([1, 50257])

# Note: Using a list [1] keeps the dimension
# vs. probas[0, 1, :] which would give shape (50257,)
```

#### Example 4: Repeated Indices

```python
# Get first token twice
repeated = probas[0, [0, 0], :]
print(repeated.shape)  # torch.Size([2, 50257])

# Both rows are identical (same token position)
```

---

## Practical Examples

### Example 1: Extract Probabilities for Specific Tokens

```python
# Setup
inputs = torch.tensor([[16833, 3626, 6100],   # "every effort moves"
                       [40, 1107, 588]])       # "I really like"

targets = torch.tensor([[3626, 6100, 345],    # " effort moves you"
                        [1107, 588, 11311]])   # " really like chocolate"

# Get model predictions
with torch.no_grad():
    logits = model(inputs)

probas = torch.softmax(logits, dim=-1)
print(probas.shape)  # torch.Size([2, 3, 50257])
```

### Example 2: Analyze First Training Example

```python
# Get all token probabilities for first example
batch_0_probs = probas[0, [0,1,2], :]  # or simply probas[0]
print(batch_0_probs.shape)  # torch.Size([3, 50257])

# For each token, find the most likely prediction
for i in range(3):
    token_probs = batch_0_probs[i]
    top_token_id = torch.argmax(token_probs)
    top_prob = token_probs[top_token_id]
    
    print(f"Position {i}: Most likely token = {top_token_id.item()}, "
          f"Probability = {top_prob.item():.4f}")
```

**Output:**
```
Position 0: Most likely token = 262, Probability = 0.0023
Position 1: Most likely token = 198, Probability = 0.0031
Position 2: Most likely token = 464, Probability = 0.0019
```

### Example 3: Compare Predictions vs Targets

```python
# What did the model predict?
predictions = torch.argmax(probas, dim=-1)
print("Predictions:\n", predictions)

# What should it have predicted?
print("Targets:\n", targets)

# Check accuracy
correct = (predictions == targets).float()
accuracy = correct.mean()
print(f"Token-level accuracy: {accuracy.item():.2%}")
```

---

## Target Probability Extraction

### The Critical Pattern for Loss Calculation

When training, we need the **probability assigned to the correct token** (target):

```python
# For batch 0, get probabilities for target tokens
text_idx = 0
target_probas_1 = probas[text_idx, [0,1,2], targets[text_idx]]
#                        ↑         ↑         ↑
#                     Batch 0    Positions  Target IDs

print(target_probas_1)
# tensor([0.0002, 0.0003, 0.0001])  # Very low! Model not trained yet
```

### Understanding the Triple Indexing

```python
probas[text_idx, [0,1,2], targets[text_idx]]
```

This is **three-dimensional indexing**:

1. **First index (`text_idx`)**: Which batch (scalar)
2. **Second index (`[0,1,2]`)**: Which token positions (list)
3. **Third index (`targets[text_idx]`)**: Which vocabulary IDs (tensor)

**What this extracts:**

```python
targets[0] = tensor([3626, 6100, 345])

# Extracts:
probas[0, 0, 3626]  # Probability of token 3626 at position 0
probas[0, 1, 6100]  # Probability of token 6100 at position 1
probas[0, 2, 345]   # Probability of token 345 at position 2
```

**Visual:**

```
Position 0:
probas[0, 0, :] = [p₀, p₁, ..., p₃₆₂₆, ..., p₅₀₂₅₆]
                                 ↑
                          Extract this! (target is 3626)

Position 1:
probas[0, 1, :] = [p₀, p₁, ..., p₆₁₀₀, ..., p₅₀₂₅₆]
                                 ↑
                          Extract this! (target is 6100)

Position 2:
probas[0, 2, :] = [p₀, p₁, ..., p₃₄₅, ..., p₅₀₂₅₆]
                                ↑
                          Extract this! (target is 345)
```

### Complete Example

```python
# Extract target probabilities for both batches
text_idx = 0
target_probas_1 = probas[text_idx, [0,1,2], targets[text_idx]]
print("Batch 0 target probabilities:", target_probas_1)

text_idx = 1
target_probas_2 = probas[text_idx, [0,1,2], targets[text_idx]]
print("Batch 1 target probabilities:", target_probas_2)

# Combine and compute log probabilities
all_target_probas = torch.cat((target_probas_1, target_probas_2))
log_probas = torch.log(all_target_probas)
print("Log probabilities:", log_probas)

# Average log probability (closer to 0 is better)
avg_log_prob = torch.mean(log_probas)
print(f"Average log probability: {avg_log_prob:.4f}")

# Cross-entropy loss (minimize this)
cross_entropy = -avg_log_prob
print(f"Cross-entropy loss: {cross_entropy:.4f}")
```

---

## Common Patterns

### Pattern 1: All Tokens in a Batch

```python
# These are equivalent when you have 3 tokens
probas[0, [0,1,2], :]  # Explicit
probas[0, :, :]        # Using slice
probas[0]              # Simplest
```

### Pattern 2: Specific Token Positions

```python
# First and last token only
probas[0, [0, 2], :]

# Middle token only
probas[0, [1], :]

# All but first
probas[0, [1, 2], :]
```

### Pattern 3: Multiple Batches, Specific Tokens

```python
# Can't use list for batch dimension directly
# Instead, use advanced indexing:

batch_indices = torch.tensor([0, 1, 0])
token_indices = torch.tensor([0, 1, 2])
vocab_index = 100

# Extract probas[0,0,100], probas[1,1,100], probas[0,2,100]
selected = probas[batch_indices, token_indices, vocab_index]
```

### Pattern 4: Broadcasting with Targets

```python
# Most common pattern in training
batch_idx = 0
token_positions = [0, 1, 2]
target_token_ids = targets[batch_idx]

# Get probability of correct token at each position
target_probs = probas[batch_idx, token_positions, target_token_ids]
```

---

## Equivalence Examples

### When Lists Match Slices

```python
# For 3 tokens, these are all equivalent:
a = probas[0, [0,1,2], :]
b = probas[0, 0:3, :]
c = probas[0, :, :]
d = probas[0]

print(torch.equal(a, b))  # True
print(torch.equal(a, c))  # True
print(torch.equal(a, d))  # True
```

### When Lists Differ from Slices

```python
# These are NOT equivalent:
a = probas[0, [0, 2], :]      # Skip middle token
b = probas[0, 0:2, :]          # First two tokens only

print(a.shape)  # torch.Size([2, 50257])
print(b.shape)  # torch.Size([2, 50257])
print(torch.equal(a, b))  # False - different tokens selected
```

---

## Performance Considerations

### List vs Slice Performance

```python
import time

# Slice indexing (contiguous memory)
start = time.time()
for _ in range(1000):
    result = probas[:, 0:3, :]
slice_time = time.time() - start

# List indexing (may require gathering)
start = time.time()
for _ in range(1000):
    result = probas[:, [0,1,2], :]
list_time = time.time() - start

print(f"Slice time: {slice_time:.4f}s")
print(f"List time: {list_time:.4f}s")
# Slices are typically 2-5x faster for contiguous access
```

### Optimization Tips

1. **Use slices when possible** for contiguous ranges:
   ```python
   # ✅ Fast (contiguous memory access)
   probas[0, :3, :]
   
   # ❌ Slower (gathering operation)
   probas[0, [0,1,2], :]
   ```

2. **Avoid repeated indexing** in loops:
   ```python
   # ❌ Slow (indexes every iteration)
   for i in range(3):
       token_probs = probas[0, i, :]
       # process...
   
   # ✅ Fast (index once)
   all_token_probs = probas[0]  # Shape: (3, 50257)
   for i in range(3):
       token_probs = all_token_probs[i]
       # process...
   ```

3. **Use tensor indexing** instead of loops when possible:
   ```python
   # ❌ Slow (Python loop)
   target_probs = []
   for i in range(3):
       target_probs.append(probas[0, i, targets[0, i]])
   
   # ✅ Fast (vectorized)
   target_probs = probas[0, [0,1,2], targets[0]]
   ```

---

## Common Mistakes and Solutions

### Mistake 1: Shape Mismatch

```python
# ❌ Wrong: Using scalar instead of list
probas[0, 1, :]  # Shape: (50257,) - lost dimension

# ✅ Correct: Use list to keep dimension
probas[0, [1], :]  # Shape: (1, 50257) - dimension preserved
```

### Mistake 2: Incorrect Target Indexing

```python
targets = torch.tensor([[3626, 6100, 345]])

# ❌ Wrong: This tries to index with entire tensor
probas[0, :, targets]  # Error or wrong shape

# ✅ Correct: Flatten or index properly
probas[0, [0,1,2], targets[0]]
```

### Mistake 3: Mixed Indexing Types

```python
# ❌ Confusing: Mixing list and slice
probas[0, [0,1], 0:1000]  # Works but unclear intent

# ✅ Clear: Be consistent
probas[0, [0,1], :]  # Get specific tokens, all vocab
```

### Mistake 4: Out of Bounds

```python
# If probas has shape (2, 3, 50257)
probas[0, [0,1,2,3], :]  # ❌ Error! Only 3 token positions
probas[0, [0,1,2], :]    # ✅ Correct
```

---

## Advanced Use Cases

### Use Case 1: Attention Score Analysis

```python
# Analyze which tokens the model focuses on
attention_weights = model.transformer_blocks[0].attn.attn_weights

# Get attention from token 2 to all other tokens
attention_from_token_2 = attention_weights[0, :, 2, :]
print(f"Attention from token 2: {attention_from_token_2.shape}")
```

### Use Case 2: Top-K Predictions

```python
# Get top 5 predictions for each token
k = 5
top_k_probs, top_k_indices = torch.topk(probas[0], k, dim=-1)

print(f"Top {k} probabilities shape: {top_k_probs.shape}")  # (3, 5)
print(f"Top {k} token IDs shape: {top_k_indices.shape}")    # (3, 5)

# Decode top predictions for first token
for i in range(k):
    token_id = top_k_indices[0, i].item()
    prob = top_k_probs[0, i].item()
    print(f"Rank {i+1}: Token {token_id}, Probability {prob:.4f}")
```

### Use Case 3: Batch Processing with Different Lengths

```python
# When sequences have different lengths (using padding)
batch_lengths = torch.tensor([3, 2])  # Actual lengths before padding

# Extract valid probabilities only
valid_probs = []
for i in range(len(batch_lengths)):
    length = batch_lengths[i].item()
    valid_probs.append(probas[i, :length, :])

# Process without including padding tokens
```

---

## Summary

### Key Takeaways

1. **List indexing** allows selecting specific, non-contiguous elements
2. **`probas[0, [0,1,2], :]`** extracts all token probabilities from batch 0
3. **Triple indexing** `probas[batch, positions, targets]` extracts target probabilities
4. **Slices are faster** than lists for contiguous access
5. **List indexing enables**:
   - Non-contiguous selection: `[0, 2]`
   - Reordering: `[2, 1, 0]`
   - Repetition: `[0, 0, 1]`

### Quick Reference

| Pattern | Purpose | Result Shape |
|---------|---------|--------------|
| `probas[0, [0,1,2], :]` | All tokens, batch 0 | `(3, 50257)` |
| `probas[0, [0, 2], :]` | First & last token | `(2, 50257)` |
| `probas[0, [1], :]` | Middle token only | `(1, 50257)` |
| `probas[0, :, :]` | Same as `[0,1,2]` | `(3, 50257)` |
| `probas[0]` | Simplest form | `(3, 50257)` |
| `probas[:, 0, :]` | First token, all batches | `(2, 50257)` |
| `probas[0, [0,1,2], targets[0]]` | Target probabilities | `(3,)` |

### When to Use What

- **Slices (`:`)**  → Contiguous ranges, better performance
- **Lists (`[0,1,2]`)** → Non-contiguous, reordering, or explicit clarity
- **Mix of both** → Extract specific dimensions while keeping others

This indexing pattern is fundamental to extracting target probabilities for loss calculation, which is essential for training GPT models!
