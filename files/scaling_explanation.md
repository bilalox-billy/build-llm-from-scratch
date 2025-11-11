# Understanding Attention Scaling in Self-Attention Mechanism

## Cell 24: The Scaling Problem - Why Large Numbers Break Softmax

This cell demonstrates **why we need to scale** attention scores before applying softmax.

### The Code

```python
import torch

# Define the tensor
tensor = torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5])

# Apply softmax without scaling
softmax_result = torch.softmax(tensor, dim=-1)
print("Softmax without scaling:", softmax_result)

# Multiply the tensor by 8 and then apply softmax
scaled_tensor = tensor * 8
softmax_scaled_result = torch.softmax(scaled_tensor, dim=-1)
print("Softmax after scaling (tensor * 8):", softmax_scaled_result)
```

### What Happens?

Think of this as 5 different attention scores before softmax.

**Without scaling:**
```
Input: [0.1, -0.2, 0.3, -0.2, 0.5]
Output: [0.19, 0.14, 0.24, 0.14, 0.29]
```
The probabilities are fairly distributed - no single value dominates.

**With scaling (multiplied by 8):**
```
Input: [0.8, -1.6, 2.4, -1.6, 4.0]
Output: [0.04, 0.00, 0.12, 0.00, 0.83]
```

### The Problem

When you multiply by 8 (making numbers bigger), softmax becomes **very confident** about the highest value (0.83 vs 0.29). It almost ignores the other values!

### Real-World Analogy

Imagine grading students:
- **Original scores:** 10, 8, 12, 8, 15 
  - After softmax, everyone gets a decent share of attention
- **Multiply by 8:** 80, 64, 96, 64, 120 
  - After softmax, only the top student (120) gets almost all the attention, others are ignored

This makes learning **unstable** because the model becomes too confident too quickly. The gradients become very small for most positions, slowing down or preventing learning.

---

## Cell 25: Why Divide by √(dimension)?

This cell demonstrates that **dot products get bigger as dimension increases**, and dividing by √(dimension) fixes it.

### The Code

```python
import numpy as np

def compute_variance(dim, num_trials=1000):
    dot_products = []
    scaled_dot_products = []

    # Generate multiple random vectors and compute dot products
    for _ in range(num_trials):
        q = np.random.randn(dim)  # Random query vector
        k = np.random.randn(dim)  # Random key vector
        
        # Compute dot product (this is what attention does)
        dot_product = np.dot(q, k)
        dot_products.append(dot_product)
        
        # Scale the dot product by sqrt(dim)
        scaled_dot_product = dot_product / np.sqrt(dim)
        scaled_dot_products.append(scaled_dot_product)
    
    # Calculate variance of the dot products
    variance_before_scaling = np.var(dot_products)
    variance_after_scaling = np.var(scaled_dot_products)

    return variance_before_scaling, variance_after_scaling

# Test with different dimensions
variance_before_5, variance_after_5 = compute_variance(5)
variance_before_100, variance_after_100 = compute_variance(100)
```

### The Magic

When `dim=5`:
- Variance before scaling: **~5**
- Variance after scaling: **~1**

When `dim=100`:
- Variance before scaling: **~100**
- Variance after scaling: **~1**

### Real-World Analogy

Imagine you're adding up random positive and negative numbers:
- **Add 5 numbers:** Total might be around -3 to +3
- **Add 100 numbers:** Total might be around -30 to +30 (much bigger range!)

When you divide by √5 or √100, you bring both ranges back to roughly -1 to +1.

### Why This Matters for Attention

In transformers, embedding dimensions can be **512, 768, or even larger**. Without scaling:
- Small dimension (d=5): dot products are manageable (variance ~5)
- Large dimension (d=512): dot products become HUGE (variance ~512)

Dividing by √512 ≈ 22.6 brings the variance back to ~1, keeping the numbers stable for softmax.

### Mathematical Insight

When you have two independent random variables with variance 1:
- Their product has variance ≈ 1
- Sum of d products has variance ≈ d
- Dividing by √d gives variance ≈ d/d = 1

This is why we use √d specifically - it's the exact value needed to normalize the variance back to 1.

---

## Summary

### Cell 24: The Softmax Sensitivity Problem
- **Problem:** Big numbers make softmax too confident → unstable learning
- **Example:** Multiplying by 8 makes the distribution "peaky" (0.83 vs 0.29)
- **Why it matters:** Gradients vanish for most positions, learning becomes difficult

### Cell 25: The Dimension Growth Problem
- **Problem:** Higher dimensions create bigger dot products (variance grows with d)
- **Solution:** Divide by √d to keep variance stable at ~1
- **Why it matters:** Keeps attention scores in a reasonable range regardless of embedding dimension

### The Complete Attention Formula

This is why the attention formula is:

```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

The `/√d_k` term is crucial for:
1. Preventing softmax from becoming too confident (Cell 24)
2. Keeping variance stable across different embedding dimensions (Cell 25)

Without this scaling, transformer models wouldn't train well, especially with large embedding dimensions!
