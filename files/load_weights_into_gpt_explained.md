# Understanding `load_weights_into_gpt()` Function

## Overview

The `load_weights_into_gpt()` function is the **master weight loader** that transfers all pretrained weights from OpenAI's GPT-2 model into your custom PyTorch `GPTModel` instance. It systematically copies **124 million parameters** (for GPT-2 small) from the downloaded checkpoint into your model.

```python
def load_weights_into_gpt(gpt, params):
    # Loads all pretrained weights from params dictionary into gpt model
    ...
```

---

## Why Do We Need This Function?

After downloading OpenAI's GPT-2 weights, you have:

1. **`params` dictionary**: Contains all pretrained weights as NumPy arrays
2. **`gpt` model**: Your GPTModel instance with random weights

The `load_weights_into_gpt()` function **bridges** these two, transforming your untrained model into OpenAI's trained GPT-2 model.

### Before vs. After

```
BEFORE load_weights_into_gpt():
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Your GPTModel (Random Weights)
├─ tok_emb.weight: [random values] (50257, 768)
├─ pos_emb.weight: [random values] (1024, 768)
├─ transformer_blocks[0]:
│  ├─ attn.W_query.weight: [random] (768, 768)
│  ├─ attn.W_key.weight: [random] (768, 768)
│  └─ ...
└─ Can't generate coherent text ❌
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AFTER load_weights_into_gpt():
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Your GPTModel (OpenAI Pretrained Weights)
├─ tok_emb.weight: [pretrained] (50257, 768) ✅
├─ pos_emb.weight: [pretrained] (1024, 768) ✅
├─ transformer_blocks[0]:
│  ├─ attn.W_query.weight: [pretrained] (768, 768) ✅
│  ├─ attn.W_key.weight: [pretrained] (768, 768) ✅
│  └─ ...
└─ Can generate coherent text ✅
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Function Parameters

```python
def load_weights_into_gpt(gpt, params):
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `gpt` | `GPTModel` | Your PyTorch model instance (target) |
| `params` | `dict` | Dictionary containing pretrained weights from OpenAI (source) |

### Understanding the `params` Dictionary Structure

```python
params = {
    'wte': np.array(...),      # Token embeddings (50257, 768)
    'wpe': np.array(...),      # Positional embeddings (1024, 768)
    'blocks': [                # List of 12 transformer blocks
        {
            'attn': {
                'c_attn': {
                    'w': np.array(...),  # Combined Q,K,V weights (768, 2304)
                    'b': np.array(...)   # Combined Q,K,V biases (2304,)
                },
                'c_proj': {
                    'w': np.array(...),  # Output projection weights (768, 768)
                    'b': np.array(...)   # Output projection bias (768,)
                }
            },
            'mlp': {
                'c_fc': {
                    'w': np.array(...),  # Feed-forward expansion (768, 3072)
                    'b': np.array(...)   # Feed-forward expansion bias (3072,)
                },
                'c_proj': {
                    'w': np.array(...),  # Feed-forward contraction (3072, 768)
                    'b': np.array(...)   # Feed-forward contraction bias (768,)
                }
            },
            'ln_1': {
                'g': np.array(...),      # LayerNorm 1 scale (768,)
                'b': np.array(...)       # LayerNorm 1 shift (768,)
            },
            'ln_2': {
                'g': np.array(...),      # LayerNorm 2 scale (768,)
                'b': np.array(...)       # LayerNorm 2 shift (768,)
            }
        },
        # ... 11 more blocks ...
    ],
    'g': np.array(...),        # Final LayerNorm scale (768,)
    'b': np.array(...)         # Final LayerNorm shift (768,)
}
```

---

## Detailed Parameter Explanations with Examples

### 1. Top-Level Parameters

#### `params['wte']` - Token Embeddings (50257, 768)

**What it means:**
- **50257**: Total vocabulary size (number of unique tokens GPT-2 can understand)
- **768**: Embedding dimension (how many features represent each token)

**Concrete example:**
```python
# Access token embeddings
print(params['wte'].shape)  # (50257, 768)

# Get embedding for token ID 0 (usually "!")
token_0_embedding = params['wte'][0]
print(token_0_embedding.shape)  # (768,)
print(token_0_embedding[:5])  
# Example output: [0.0234, -0.0567, 0.0891, -0.1234, 0.0456]
# These 768 numbers represent the meaning of token "!"

# Get embedding for token ID 15496 (word "Hello")
hello_embedding = params['wte'][15496]
print(hello_embedding[:5])
# Example output: [-0.0891, 0.1234, -0.0567, 0.0789, -0.0345]
# Different numbers = different meaning
```

**Visual representation:**
```
params['wte'] = Array with 50,257 rows (tokens) × 768 columns (features)
┌─────────────────────────────────────────────────────────────┐
│ Token 0:     [0.023, -0.056, 0.089, ..., 0.045]  768 nums  │
│ Token 1:     [-0.012, 0.078, -0.034, ..., -0.067] 768 nums │
│ Token 2:     [0.045, -0.089, 0.123, ..., 0.012]  768 nums  │
│ ...                                                          │
│ Token 15496: [-0.089, 0.123, -0.056, ..., -0.034] "Hello"  │
│ Token 15497: [0.067, -0.123, 0.089, ..., 0.045]  "World"   │
│ ...                                                          │
│ Token 50256: [0.012, -0.045, 0.078, ..., -0.023] Last token│
└─────────────────────────────────────────────────────────────┘
```

---

#### `params['wpe']` - Positional Embeddings (1024, 768)

**What it means:**
- **1024**: Maximum sequence length (max tokens GPT-2 can process at once)
- **768**: Embedding dimension (same as token embeddings)

**Concrete example:**
```python
# Access positional embeddings
print(params['wpe'].shape)  # (1024, 768)

# Get embedding for position 0 (first token in sequence)
position_0_embedding = params['wpe'][0]
print(position_0_embedding[:5])
# Example: [0.0123, -0.0456, 0.0789, -0.0234, 0.0567]

# Get embedding for position 5 (sixth token in sequence)
position_5_embedding = params['wpe'][5]
print(position_5_embedding[:5])
# Example: [-0.0345, 0.0678, -0.0912, 0.0234, -0.0456]
# Different position = different encoding
```

**How it's used:**
```python
# For input "Hello world" (tokens [15496, 995]):
token_emb = params['wte'][15496] + params['wte'][995]  # Token meanings
pos_emb = params['wpe'][0] + params['wpe'][1]          # Position info

# Final input = token_emb + pos_emb
# This tells the model WHAT the tokens are AND WHERE they appear
```

---

### 2. Transformer Block Parameters (`params['blocks'][0]`)

Each of the 12 transformer blocks contains identical structure. Let's examine block 0:

---

#### `params['blocks'][0]['attn']['c_attn']['w']` - Combined Q,K,V Weights (768, 2304)

**What it means:**
- **768**: Input dimension (embedding size)
- **2304**: Output dimension = 768 × 3 (Query + Key + Value concatenated)

**Why 2304?**
```
2304 = 768 (Query) + 768 (Key) + 768 (Value)
```

**Concrete example:**
```python
# This single matrix contains 3 weight matrices stacked together
combined_weights = params['blocks'][0]['attn']['c_attn']['w']
print(combined_weights.shape)  # (768, 2304)

# Split into 3 separate matrices:
query_weights = combined_weights[:, 0:768]      # First 768 columns
key_weights = combined_weights[:, 768:1536]     # Middle 768 columns  
value_weights = combined_weights[:, 1536:2304]  # Last 768 columns

print(query_weights.shape)  # (768, 768)
print(key_weights.shape)    # (768, 768)
print(value_weights.shape)  # (768, 768)

# Example values from first row:
print(combined_weights[0, :5])      # Query part: [0.023, -0.056, ...]
print(combined_weights[0, 768:773]) # Key part:   [-0.034, 0.078, ...]
print(combined_weights[0, 1536:1541]) # Value part: [0.045, -0.089, ...]
```

**Visual representation:**
```
combined_weights (768, 2304):
┌─────────────────┬─────────────────┬─────────────────┐
│  Query weights  │   Key weights   │  Value weights  │
│   (768, 768)    │   (768, 768)    │   (768, 768)    │
│                 │                 │                 │
│  columns 0-767  │  cols 768-1535  │  cols 1536-2303 │
└─────────────────┴─────────────────┴─────────────────┘
        ↓                 ↓                 ↓
    Used to           Used to           Used to
  compute Q         compute K         compute V
  (queries)         (keys)            (values)
```

**How it's used:**
```python
# Input embedding (one token):
x = np.array([0.1, 0.2, ..., 0.8])  # Shape: (768,)

# Compute Q, K, V by matrix multiplication:
Q = x @ query_weights   # (768,) @ (768, 768) = (768,)
K = x @ key_weights     # (768,) @ (768, 768) = (768,)
V = x @ value_weights   # (768,) @ (768, 768) = (768,)

# These Q, K, V are then used in attention mechanism
```

---

#### `params['blocks'][0]['attn']['c_attn']['b']` - Combined Q,K,V Biases (2304,)

**What it means:**
- **2304**: Bias values = 768 (Query) + 768 (Key) + 768 (Value)

**Concrete example:**
```python
# Single bias vector for all Q, K, V
combined_bias = params['blocks'][0]['attn']['c_attn']['b']
print(combined_bias.shape)  # (2304,)

# Split into 3 parts:
query_bias = combined_bias[0:768]      # First 768 values
key_bias = combined_bias[768:1536]     # Middle 768 values
value_bias = combined_bias[1536:2304]  # Last 768 values

print(query_bias.shape)  # (768,)
print(key_bias.shape)    # (768,)
print(value_bias.shape)  # (768,)

# Example values:
print(query_bias[:5])  # [0.001, -0.002, 0.003, -0.001, 0.002]
print(key_bias[:5])    # [-0.002, 0.004, -0.001, 0.003, -0.002]
print(value_bias[:5])  # [0.003, -0.001, 0.002, -0.004, 0.001]
```

**How it's used:**
```python
# Computing Q with bias:
Q = (x @ query_weights) + query_bias
#   (matrix multiply)   + (add bias)
#   (768,)             + (768,)  = (768,)
```

---

#### `params['blocks'][0]['attn']['c_proj']['w']` - Attention Output Projection (768, 768)

**What it means:**
- **First 768**: Input dimension (concatenated multi-head output)
- **Second 768**: Output dimension (back to embedding size)

**Concrete example:**
```python
output_proj_weights = params['blocks'][0]['attn']['c_proj']['w']
print(output_proj_weights.shape)  # (768, 768)

# Example: First row, first 5 columns
print(output_proj_weights[0, :5])
# [0.0234, -0.0567, 0.0891, -0.0123, 0.0456]

# This is a square matrix that transforms the attention output
```

**How it's used:**
```python
# After multi-head attention produces output:
attention_output = np.array([...])  # Shape: (768,)

# Project to final output:
final_output = attention_output @ output_proj_weights + output_proj_bias
#              (768,) @ (768, 768) = (768,)
```

---

#### `params['blocks'][0]['attn']['c_proj']['b']` - Output Projection Bias (768,)

**What it means:**
- **768**: One bias value for each output dimension

**Concrete example:**
```python
output_proj_bias = params['blocks'][0]['attn']['c_proj']['b']
print(output_proj_bias.shape)  # (768,)
print(output_proj_bias[:10])
# [0.001, -0.002, 0.003, -0.001, 0.002, -0.003, 0.001, -0.002, 0.004, -0.001]
```

---

### 3. Feed-Forward Network (MLP) Parameters

#### `params['blocks'][0]['mlp']['c_fc']['w']` - Feed-Forward Expansion (768, 3072)

**What it means:**
- **768**: Input dimension (embedding size)
- **3072**: Output dimension = 768 × 4 (4x expansion)

**Why 3072?**
```
3072 = 768 × 4
GPT-2 expands embeddings by 4x in the feed-forward network
```

**Concrete example:**
```python
ff_expansion = params['blocks'][0]['mlp']['c_fc']['w']
print(ff_expansion.shape)  # (768, 3072)

# Example: First row shows how first input feature contributes
print(ff_expansion[0, :5])  
# [0.023, -0.056, 0.089, -0.012, 0.045]

# Example: Column 0 shows how all inputs contribute to first output
print(ff_expansion[:5, 0])
# [0.023, -0.034, 0.067, -0.089, 0.012]
```

**How it's used:**
```python
# Input from attention layer:
x = np.array([...])  # Shape: (768,)

# Expand to 3072 dimensions:
expanded = x @ ff_expansion + ff_expansion_bias
#         (768,) @ (768, 3072) = (3072,)
#         More dimensions = more capacity to learn patterns!

# Then apply GELU activation:
activated = gelu(expanded)  # Still (3072,)
```

**Visual representation:**
```
Input        Expansion Matrix        Output
(768,)   ×   (768, 3072)      =     (3072,)
┌───┐       ┌─────────────┐        ┌─────┐
│ x │   @   │ Each of 768 │   =    │ 4x  │
│   │       │ inputs gets │        │more │
│   │       │ mapped to   │        │dims │
│   │       │ 3072 outputs│        │     │
└───┘       └─────────────┘        └─────┘
```

---

#### `params['blocks'][0]['mlp']['c_fc']['b']` - Expansion Bias (3072,)

**What it means:**
- **3072**: One bias for each expanded dimension

**Concrete example:**
```python
ff_expansion_bias = params['blocks'][0]['mlp']['c_fc']['b']
print(ff_expansion_bias.shape)  # (3072,)
print(ff_expansion_bias[:10])
# [0.001, -0.002, 0.003, -0.001, 0.002, -0.003, 0.001, -0.002, 0.004, -0.001]
```

---

#### `params['blocks'][0]['mlp']['c_proj']['w']` - Feed-Forward Contraction (3072, 768)

**What it means:**
- **3072**: Input dimension (expanded size)
- **768**: Output dimension (back to embedding size)

**Concrete example:**
```python
ff_contraction = params['blocks'][0]['mlp']['c_proj']['w']
print(ff_contraction.shape)  # (3072, 768)

# This is the OPPOSITE of expansion - shrinks back down
print(ff_contraction[:5, 0])
# [0.012, -0.034, 0.056, -0.023, 0.045]
```

**How it's used:**
```python
# After GELU activation (3072 dimensions):
activated = np.array([...])  # Shape: (3072,)

# Contract back to 768 dimensions:
output = activated @ ff_contraction + ff_contraction_bias
#       (3072,) @ (3072, 768) = (768,)
#       Back to original size!
```

**Visual representation:**
```
Input         Contraction Matrix      Output
(3072,)   ×   (3072, 768)      =     (768,)
┌─────┐      ┌─────────────┐        ┌───┐
│ 4x  │  @   │ Each of 3072│   =    │ x │
│more │      │ inputs gets │        │   │
│dims │      │ mapped to   │        │   │
│     │      │ 768 outputs │        │   │
└─────┘      └─────────────┘        └───┘
```

---

#### `params['blocks'][0]['mlp']['c_proj']['b']` - Contraction Bias (768,)

**What it means:**
- **768**: One bias for each output dimension

**Concrete example:**
```python
ff_contraction_bias = params['blocks'][0]['mlp']['c_proj']['b']
print(ff_contraction_bias.shape)  # (768,)
print(ff_contraction_bias[:10])
# [0.002, -0.001, 0.003, -0.002, 0.001, -0.003, 0.002, -0.001, 0.004, -0.002]
```

---

### 4. Layer Normalization Parameters

#### `params['blocks'][0]['ln_1']['g']` - LayerNorm 1 Scale (768,)

**What it means:**
- **768**: One scale parameter for each embedding dimension
- **'g'** stands for "gamma" (scale parameter in LayerNorm)

**Concrete example:**
```python
ln1_scale = params['blocks'][0]['ln_1']['g']
print(ln1_scale.shape)  # (768,)
print(ln1_scale[:10])
# [1.012, 0.998, 1.005, 0.995, 1.008, 0.992, 1.003, 0.997, 1.010, 0.993]
# Notice values are close to 1.0 (typical for scale parameters)
```

**How it's used:**
```python
# LayerNorm formula:
# normalized = (x - mean) / sqrt(variance + epsilon)
# output = scale * normalized + shift

x = np.array([...])  # Shape: (768,)
mean = x.mean()
var = x.var()

normalized = (x - mean) / np.sqrt(var + 1e-5)
output = ln1_scale * normalized + ln1_shift
#        (768,) * (768,)     + (768,)
```

---

#### `params['blocks'][0]['ln_1']['b']` - LayerNorm 1 Shift (768,)

**What it means:**
- **768**: One shift parameter for each embedding dimension
- **'b'** stands for "beta" (shift parameter in LayerNorm)

**Concrete example:**
```python
ln1_shift = params['blocks'][0]['ln_1']['b']
print(ln1_shift.shape)  # (768,)
print(ln1_shift[:10])
# [0.001, -0.002, 0.003, -0.001, 0.002, -0.003, 0.001, -0.002, 0.004, -0.001]
# Small values close to 0 (typical for shift parameters)
```

---

#### `params['blocks'][0]['ln_2']['g']` and `params['blocks'][0]['ln_2']['b']`

**Same as ln_1, but for the second LayerNorm** (after feed-forward network)

```python
ln2_scale = params['blocks'][0]['ln_2']['g']  # (768,)
ln2_shift = params['blocks'][0]['ln_2']['b']  # (768,)
```

---

### 5. Final Layer Normalization

#### `params['g']` - Final LayerNorm Scale (768,)

**What it means:**
- **768**: Scale parameter for the final normalization before output
- Applied after all 12 transformer blocks

**Concrete example:**
```python
final_scale = params['g']
print(final_scale.shape)  # (768,)
print(final_scale[:10])
# [1.015, 0.997, 1.008, 0.993, 1.012, 0.989, 1.006, 0.995, 1.013, 0.991]
```

---

#### `params['b']` - Final LayerNorm Shift (768,)

**What it means:**
- **768**: Shift parameter for the final normalization before output

**Concrete example:**
```python
final_shift = params['b']
print(final_shift.shape)  # (768,)
print(final_shift[:10])
# [0.002, -0.001, 0.003, -0.002, 0.001, -0.003, 0.002, -0.001, 0.004, -0.002]
```

---

## Complete Parameter Inspection Example

```python
# Load the parameters
from src.gpt_download3 import download_and_load_gpt2
settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")

# ═══════════════════════════════════════════════════════════════
# Inspect embeddings
# ═══════════════════════════════════════════════════════════════
print("Token embeddings shape:", params['wte'].shape)  # (50257, 768)
print("First token embedding (first 5 dims):", params['wte'][0, :5])

print("\nPosition embeddings shape:", params['wpe'].shape)  # (1024, 768)
print("First position embedding (first 5 dims):", params['wpe'][0, :5])

# ═══════════════════════════════════════════════════════════════
# Inspect transformer block 0
# ═══════════════════════════════════════════════════════════════
block_0 = params['blocks'][0]

print("\n=== Block 0 Attention ===")
print("Combined QKV weights:", block_0['attn']['c_attn']['w'].shape)  # (768, 2304)
print("Combined QKV bias:", block_0['attn']['c_attn']['b'].shape)     # (2304,)
print("Output projection weights:", block_0['attn']['c_proj']['w'].shape)  # (768, 768)
print("Output projection bias:", block_0['attn']['c_proj']['b'].shape)     # (768,)

print("\n=== Block 0 Feed-Forward ===")
print("Expansion weights:", block_0['mlp']['c_fc']['w'].shape)     # (768, 3072)
print("Expansion bias:", block_0['mlp']['c_fc']['b'].shape)        # (3072,)
print("Contraction weights:", block_0['mlp']['c_proj']['w'].shape) # (3072, 768)
print("Contraction bias:", block_0['mlp']['c_proj']['b'].shape)    # (768,)

print("\n=== Block 0 LayerNorms ===")
print("LayerNorm 1 scale:", block_0['ln_1']['g'].shape)  # (768,)
print("LayerNorm 1 shift:", block_0['ln_1']['b'].shape)  # (768,)
print("LayerNorm 2 scale:", block_0['ln_2']['g'].shape)  # (768,)
print("LayerNorm 2 shift:", block_0['ln_2']['b'].shape)  # (768,)

# ═══════════════════════════════════════════════════════════════
# Inspect final normalization
# ═══════════════════════════════════════════════════════════════
print("\n=== Final LayerNorm ===")
print("Final scale shape:", params['g'].shape)  # (768,)
print("Final shift shape:", params['b'].shape)  # (768,)

# ═══════════════════════════════════════════════════════════════
# Count total parameters
# ═══════════════════════════════════════════════════════════════
total = 0
total += params['wte'].size      # 50257 * 768 = 38,597,376
total += params['wpe'].size      # 1024 * 768 = 786,432

for block in params['blocks']:   # 12 blocks
    total += block['attn']['c_attn']['w'].size  # 768 * 2304
    total += block['attn']['c_attn']['b'].size  # 2304
    total += block['attn']['c_proj']['w'].size  # 768 * 768
    total += block['attn']['c_proj']['b'].size  # 768
    total += block['mlp']['c_fc']['w'].size     # 768 * 3072
    total += block['mlp']['c_fc']['b'].size     # 3072
    total += block['mlp']['c_proj']['w'].size   # 3072 * 768
    total += block['mlp']['c_proj']['b'].size   # 768
    total += block['ln_1']['g'].size            # 768
    total += block['ln_1']['b'].size            # 768
    total += block['ln_2']['g'].size            # 768
    total += block['ln_2']['b'].size            # 768

total += params['g'].size  # 768
total += params['b'].size  # 768

print(f"\nTotal parameters: {total:,}")  # 124,439,808
```

---

## Summary Table: All Parameters at a Glance

| Parameter | Shape | Purpose | Example Values |
|-----------|-------|---------|----------------|
| `params['wte']` | (50257, 768) | Token embeddings | Each token → 768-dim vector |
| `params['wpe']` | (1024, 768) | Position embeddings | Each position → 768-dim vector |
| **Per Block (×12):** | | | |
| `attn.c_attn.w` | (768, 2304) | Q,K,V weights concatenated | 2304 = 768×3 |
| `attn.c_attn.b` | (2304,) | Q,K,V biases concatenated | 2304 = 768+768+768 |
| `attn.c_proj.w` | (768, 768) | Attention output projection | Square matrix |
| `attn.c_proj.b` | (768,) | Output projection bias | One per dimension |
| `mlp.c_fc.w` | (768, 3072) | Feed-forward expansion | 3072 = 768×4 |
| `mlp.c_fc.b` | (3072,) | Expansion bias | 4x larger |
| `mlp.c_proj.w` | (3072, 768) | Feed-forward contraction | Back to 768 |
| `mlp.c_proj.b` | (768,) | Contraction bias | Original size |
| `ln_1.g` | (768,) | LayerNorm 1 scale | Close to 1.0 |
| `ln_1.b` | (768,) | LayerNorm 1 shift | Close to 0.0 |
| `ln_2.g` | (768,) | LayerNorm 2 scale | Close to 1.0 |
| `ln_2.b` | (768,) | LayerNorm 2 shift | Close to 0.0 |
| `params['g']` | (768,) | Final LayerNorm scale | Close to 1.0 |
| `params['b']` | (768,) | Final LayerNorm shift | Close to 0.0 |

**Total: 124,439,808 parameters** (with weight tying for output head)

---

## Complete Function Breakdown

### Section 1: Load Embedding Layers

```python
gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])
```

**What it does:**
- Loads **positional embeddings** (position → vector mapping)
- Loads **token embeddings** (token ID → vector mapping)

**Shape details:**

```python
# Positional Embeddings
params["wpe"].shape  # (1024, 768) - 1024 positions, 768-dim vectors
gpt.pos_emb.weight.shape  # torch.Size([1024, 768])

# Token Embeddings  
params["wte"].shape  # (50257, 768) - 50257 vocab size, 768-dim vectors
gpt.tok_emb.weight.shape  # torch.Size([50257, 768])
```

**Visual representation:**

```
Position Embeddings (wpe):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Position 0:    [0.123, -0.456, 0.789, ..., 0.234]  (768 dims)
Position 1:    [-0.234, 0.567, -0.890, ..., -0.123] (768 dims)
...
Position 1023: [0.456, -0.789, 0.123, ..., 0.567]  (768 dims)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Token Embeddings (wte):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Token 0 "!":     [0.234, -0.567, 0.890, ..., 0.345]  (768 dims)
Token 1 '"':     [-0.345, 0.678, -0.901, ..., -0.234] (768 dims)
...
Token 50256:     [0.567, -0.890, 0.234, ..., 0.678]  (768 dims)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Example usage:**
```python
# After loading embeddings:
input_ids = torch.tensor([[15496, 995]])  # "Hello world"
tok_emb = gpt.tok_emb(input_ids)  # Shape: (1, 2, 768)
pos_emb = gpt.pos_emb(torch.arange(2))  # Shape: (2, 768)

# Embeddings are now pretrained ✅
print(tok_emb[0, 0, :5])  # tensor([-0.0234, 0.0567, -0.0891, 0.1234, -0.0456])
```

---

### Section 2: Load Transformer Blocks (The Big Loop)

```python
for b in range(len(params["blocks"])):  # Iterate through 12 blocks
    # Load attention weights
    # Load feed-forward weights
    # Load layer norm weights
```

This loop processes **each of the 12 transformer layers**, loading all their components.

**Loop iterations:**
```
b=0  → Load transformer_blocks[0]  (Layer 1)
b=1  → Load transformer_blocks[1]  (Layer 2)
...
b=11 → Load transformer_blocks[11] (Layer 12)
```

---

### Section 2.1: Split and Load Attention Weights

```python
q_w, k_w, v_w = np.split(params["blocks"][b]["attn"]["c_attn"]["w"], 3, axis=-1)
```

**Critical concept:** OpenAI stores Q, K, V weights **concatenated** in a single matrix for efficiency.

**Shape transformation:**

```python
# OpenAI's format (concatenated):
params["blocks"][0]["attn"]["c_attn"]["w"].shape  # (768, 2304)
#                                                    ↑     ↑
#                                                  emb_dim  768*3

# After splitting:
q_w.shape  # (768, 768) - Query weights
k_w.shape  # (768, 768) - Key weights
v_w.shape  # (768, 768) - Value weights
```

**Visual representation:**

```
Original Concatenated Matrix (768, 2304):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┌───────────────┬───────────────┬───────────────┐
│  Query (768)  │   Key (768)   │  Value (768)  │
│               │               │               │
│  q_w weights  │  k_w weights  │  v_w weights  │
│               │               │               │
└───────────────┴───────────────┴───────────────┘
        ↓               ↓               ↓
     Split at         Split at         
   axis=-1 into 3 equal parts
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

After Splitting:
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│  Query (768)  │ │   Key (768)   │ │  Value (768)  │
│               │ │               │ │               │
│  q_w weights  │ │  k_w weights  │ │  v_w weights  │
│               │ │               │ │               │
└───────────────┘ └───────────────┘ └───────────────┘
    (768, 768)        (768, 768)        (768, 768)
```

**Why `.T` (transpose)?**

```python
gpt.transformer_blocks[b].attn.W_query.weight = assign(
    gpt.transformer_blocks[b].attn.W_query.weight, q_w.T)
```

OpenAI uses **different weight ordering** than PyTorch:
- **OpenAI (TensorFlow)**: Weights are `(in_features, out_features)`
- **PyTorch**: Weights are `(out_features, in_features)`

```python
# OpenAI's q_w:
q_w.shape  # (768, 768) - (in, out)

# PyTorch expects:
gpt.transformer_blocks[0].attn.W_query.weight.shape  # (768, 768) - (out, in)

# Solution: Transpose
q_w.T.shape  # (768, 768) - Now matches PyTorch convention
```

---

### Section 2.2: Load Query, Key, Value Weights

```python
gpt.transformer_blocks[b].attn.W_query.weight = assign(
    gpt.transformer_blocks[b].attn.W_query.weight, q_w.T)
gpt.transformer_blocks[b].attn.W_key.weight = assign(
    gpt.transformer_blocks[b].attn.W_key.weight, k_w.T)
gpt.transformer_blocks[b].attn.W_value.weight = assign(
    gpt.transformer_blocks[b].attn.W_value.weight, v_w.T)
```

**What it does:**
- Assigns **query projection** weights (transforms input to queries)
- Assigns **key projection** weights (transforms input to keys)
- Assigns **value projection** weights (transforms input to values)

**Complete example for block 0:**

```python
# Before loading (random weights):
print(gpt.transformer_blocks[0].attn.W_query.weight[0, :5])
# tensor([0.0234, -0.0891, 0.1532, -0.0456, 0.0789])

# Load pretrained weights:
load_weights_into_gpt(gpt, params)

# After loading (pretrained weights):
print(gpt.transformer_blocks[0].attn.W_query.weight[0, :5])
# tensor([-0.1234, 0.5678, -0.9012, 0.3456, -0.7890])
```

---

### Section 2.3: Load Attention Biases

```python
q_b, k_b, v_b = np.split(
    params["blocks"][b]["attn"]["c_attn"]["b"], 3, axis=-1)
```

**Same splitting concept** as weights, but for bias vectors:

```python
# Original concatenated bias:
params["blocks"][0]["attn"]["c_attn"]["b"].shape  # (2304,)

# After splitting:
q_b.shape  # (768,) - Query bias
k_b.shape  # (768,) - Key bias
v_b.shape  # (768,) - Value bias
```

**Visual representation:**

```
Concatenated Bias Vector (2304,):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[0.01, 0.02, ..., 0.11,  -0.05, 0.03, ..., 0.09,  0.02, -0.01, ..., 0.07]
 \_____  Query (768)  ___/  \_____ Key (768) ____/  \____ Value (768) ___/
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Then assign to model:**

```python
gpt.transformer_blocks[b].attn.W_query.bias = assign(
    gpt.transformer_blocks[b].attn.W_query.bias, q_b)
gpt.transformer_blocks[b].attn.W_key.bias = assign(
    gpt.transformer_blocks[b].attn.W_key.bias, k_b)
gpt.transformer_blocks[b].attn.W_value.bias = assign(
    gpt.transformer_blocks[b].attn.W_value.bias, v_b)
```

---

### Section 2.4: Load Attention Output Projection

```python
gpt.transformer_blocks[b].attn.out_proj.weight = assign(
    gpt.transformer_blocks[b].attn.out_proj.weight, 
    params["blocks"][b]["attn"]["c_proj"]["w"].T)

gpt.transformer_blocks[b].attn.out_proj.bias = assign(
    gpt.transformer_blocks[b].attn.out_proj.bias, 
    params["blocks"][b]["attn"]["c_proj"]["b"])
```

**What it does:**
- Loads the **output projection** that combines multi-head attention outputs
- This is the final linear layer in the attention mechanism

**Shape:**
```python
params["blocks"][0]["attn"]["c_proj"]["w"].shape  # (768, 768)
params["blocks"][0]["attn"]["c_proj"]["b"].shape  # (768,)
```

**Attention flow:**

```
Input (768-dim)
    ↓
[Q, K, V projections] → Multi-head attention → Concatenate heads
    ↓
Output projection (c_proj) ← We load these weights here
    ↓
Output (768-dim)
```

---

### Section 2.5: Load Feed-Forward Network Weights

```python
gpt.transformer_blocks[b].ff.layers[0].weight = assign(
    gpt.transformer_blocks[b].ff.layers[0].weight, 
    params["blocks"][b]["mlp"]["c_fc"]["w"].T)
gpt.transformer_blocks[b].ff.layers[0].bias = assign(
    gpt.transformer_blocks[b].ff.layers[0].bias, 
    params["blocks"][b]["mlp"]["c_fc"]["b"])
```

**What it does:**
- Loads **expansion layer** (768 → 3072 dimensions)
- This is the first linear layer in the feed-forward network

**Shape details:**

```python
# Expansion layer:
params["blocks"][0]["mlp"]["c_fc"]["w"].shape  # (768, 3072)
params["blocks"][0]["mlp"]["c_fc"]["b"].shape  # (3072,)

# After transpose:
params["blocks"][0]["mlp"]["c_fc"]["w"].T.shape  # (3072, 768)
# Matches PyTorch's (out_features, in_features) convention
```

**Visual representation:**

```
Feed-Forward Network:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input (768-dim)
    ↓
[Expansion Layer (c_fc)] ← We load these weights
    ↓
Expanded (3072-dim) - 4x larger!
    ↓
[GELU Activation]
    ↓
[Contraction Layer (c_proj)] ← We load these weights next
    ↓
Output (768-dim) - Back to original size
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Then load contraction layer:**

```python
gpt.transformer_blocks[b].ff.layers[2].weight = assign(
    gpt.transformer_blocks[b].ff.layers[2].weight, 
    params["blocks"][b]["mlp"]["c_proj"]["w"].T)
gpt.transformer_blocks[b].ff.layers[2].bias = assign(
    gpt.transformer_blocks[b].ff.layers[2].bias, 
    params["blocks"][b]["mlp"]["c_proj"]["b"])
```

**Why `layers[2]` not `layers[1]`?**

```python
# Feed-forward network structure:
self.ff = nn.Sequential(
    nn.Linear(768, 3072),  # layers[0] - Expansion (c_fc)
    GELU(),                # layers[1] - Activation (no weights)
    nn.Linear(3072, 768),  # layers[2] - Contraction (c_proj)
)
```

The GELU activation is at index 1, so contraction layer is at index 2!

**Shape details:**

```python
# Contraction layer:
params["blocks"][0]["mlp"]["c_proj"]["w"].shape  # (3072, 768)
params["blocks"][0]["mlp"]["c_proj"]["b"].shape  # (768,)
```

---

### Section 2.6: Load Layer Normalization Parameters

```python
gpt.transformer_blocks[b].norm1.scale = assign(
    gpt.transformer_blocks[b].norm1.scale, 
    params["blocks"][b]["ln_1"]["g"])
gpt.transformer_blocks[b].norm1.shift = assign(
    gpt.transformer_blocks[b].norm1.shift, 
    params["blocks"][b]["ln_1"]["b"])
```

**What it does:**
- Loads **first LayerNorm** parameters (before attention)
- `g` = scale parameter (gamma)
- `b` = shift parameter (beta)

**LayerNorm formula:**

```
output = scale * (normalized_input) + shift
         ↑                             ↑
      gamma (g)                     beta (b)
```

**Shape:**
```python
params["blocks"][0]["ln_1"]["g"].shape  # (768,)
params["blocks"][0]["ln_1"]["b"].shape  # (768,)
```

**Then load second LayerNorm:**

```python
gpt.transformer_blocks[b].norm2.scale = assign(
    gpt.transformer_blocks[b].norm2.scale, 
    params["blocks"][b]["ln_2"]["g"])
gpt.transformer_blocks[b].norm2.shift = assign(
    gpt.transformer_blocks[b].norm2.shift, 
    params["blocks"][b]["ln_2"]["b"])
```

**Transformer block structure:**

```
Input
  ↓
[LayerNorm 1 (ln_1)] ← We load scale & shift here
  ↓
[Multi-head Attention]
  ↓
[Residual Connection]
  ↓
[LayerNorm 2 (ln_2)] ← We load scale & shift here
  ↓
[Feed-Forward Network]
  ↓
[Residual Connection]
  ↓
Output
```

---

### Section 3: Load Final Layer Normalization

```python
gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
```

**What it does:**
- Loads the **final LayerNorm** after all transformer blocks
- This normalizes outputs before the final projection to vocabulary

**Model flow:**

```
Input tokens
    ↓
Token + Position Embeddings
    ↓
[Transformer Block 1]
[Transformer Block 2]
...
[Transformer Block 12]
    ↓
[Final LayerNorm] ← We load these parameters
    ↓
[Output Head] → Logits for each vocab token
```

**Shape:**
```python
params["g"].shape  # (768,) - Scale for 768-dim vectors
params["b"].shape  # (768,) - Shift for 768-dim vectors
```

---

### Section 4: Weight Tying (Reuse Token Embeddings)

```python
gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
```

**Critical concept:** OpenAI uses **weight tying** to reduce parameters.

**What is weight tying?**
- The output projection layer **reuses** the token embedding weights
- Instead of learning separate output weights, it uses transposed embeddings

**Why do this?**

1. **Reduces parameters**: Saves ~38M parameters (50257 × 768)
2. **Improves performance**: Embeddings and outputs share knowledge
3. **Efficient**: Same semantic space for inputs and outputs

**Visual representation:**

```
WITHOUT Weight Tying (More Parameters):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
tok_emb.weight:  (50257, 768) - 38,596,896 parameters
out_head.weight: (50257, 768) - 38,596,896 parameters
Total: 77,193,792 parameters
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WITH Weight Tying (Shared Parameters):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
tok_emb.weight:  (50257, 768) - 38,596,896 parameters
out_head.weight:  (50257, 768) - SHARES tok_emb.weight ✅
Total: 38,596,896 parameters (SAVED ~38M parameters!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**How it works:**

```python
# After loading:
print(torch.equal(gpt.tok_emb.weight, gpt.out_head.weight))
# True - They share the same weights!

# Forward pass:
logits = gpt(input_ids)  # Shape: (batch, seq_len, vocab_size)
# logits = hidden_states @ tok_emb.weight.T
#        = (batch, seq_len, 768) @ (50257, 768).T
#        = (batch, seq_len, 50257)
```

---

## Complete Parameter Count

After loading, your model has exactly **124,439,808 parameters**:

```python
total_params = sum(p.numel() for p in gpt.parameters())
print(f"Total parameters: {total_params:,}")
# Total parameters: 124,439,808
```

**Breakdown by component:**

| Component | Parameters | Calculation |
|-----------|------------|-------------|
| Token Embeddings | 38,597,376 | 50,257 × 768 |
| Position Embeddings | 786,432 | 1,024 × 768 |
| Transformer Blocks (×12) | 84,934,656 | See below |
| Final LayerNorm | 1,536 | 768 × 2 |
| Output Head | 0 | Shares token embeddings |
| **Total** | **124,439,808** | |

**Per transformer block (×12):**

| Sub-component | Parameters | Calculation |
|---------------|------------|-------------|
| Attention Q,K,V | 1,769,472 | (768 × 768 + 768) × 3 |
| Attention Output | 590,592 | 768 × 768 + 768 |
| Feed-Forward 1 | 2,362,368 | 768 × 3,072 + 3,072 |
| Feed-Forward 2 | 2,360,064 | 3,072 × 768 + 768 |
| LayerNorm 1 | 1,536 | 768 × 2 |
| LayerNorm 2 | 1,536 | 768 × 2 |
| **Per Block Total** | **7,085,568** | |
| **All 12 Blocks** | **85,026,816** | 7,085,568 × 12 |

---

## Common Errors and Solutions

### Error 1: Attribute Error `'attn' has no attribute 'W_query'`

```python
AttributeError: 'MultiHeadAttention' object has no attribute 'W_query'
```

**Cause:** Typo in the function - using `att` instead of `attn`

**Solution:**
```python
# WRONG:
gpt.transformer_blocks[b].att.W_query.weight = assign(...)

# CORRECT:
gpt.transformer_blocks[b].attn.W_query.weight = assign(...)
```

---

### Error 2: Shape Mismatch

```python
ValueError: Shape mismatch: left torch.Size([768, 768]), right (1024, 1024)
```

**Cause:** Model configuration doesn't match pretrained weights

**Solution:** Ensure config matches OpenAI's model:
```python
NEW_CONFIG = {
    "vocab_size": 50257,    # Must match
    "context_length": 1024, # Must match
    "emb_dim": 768,         # Must match
    "n_heads": 12,          # Must match
    "n_layers": 12,         # Must match
    "qkv_bias": True        # Must be True for OpenAI weights
}
```

---

### Error 3: Missing `.T` (Transpose)

```python
# Without transpose:
gpt.transformer_blocks[0].attn.W_query.weight = assign(
    gpt.transformer_blocks[0].attn.W_query.weight, q_w)
# ValueError: Shape mismatch: left torch.Size([768, 768]), right (768, 768)
```

**Why this happens:**
- Both shapes are (768, 768), but **dimensions mean different things**
- OpenAI: (in_features, out_features)
- PyTorch: (out_features, in_features)

**Solution:** Always transpose OpenAI weights:
```python
gpt.transformer_blocks[0].attn.W_query.weight = assign(
    gpt.transformer_blocks[0].attn.W_query.weight, q_w.T)  # Add .T
```

---

### Error 4: Wrong Layer Index for Feed-Forward

```python
# WRONG - GELU has no weights:
gpt.transformer_blocks[b].ff.layers[1].weight = assign(...)
# AttributeError: 'GELU' object has no attribute 'weight'

# CORRECT - Skip GELU, use layers[2]:
gpt.transformer_blocks[b].ff.layers[2].weight = assign(...)
```

---

## Testing the Loaded Weights

```python
import torch
import torch.nn as nn

# ═══════════════════════════════════════════════════════════════
# Step 1: Download and load OpenAI weights
# ═══════════════════════════════════════════════════════════════

from src.gpt_download3 import download_and_load_gpt2

settings, params = download_and_load_gpt2(
    model_size="124M",
    models_dir="gpt2"
)

print("Loaded settings:", settings)
print("Params keys:", params.keys())

# ═══════════════════════════════════════════════════════════════
# Step 2: Create model with matching configuration
# ═══════════════════════════════════════════════════════════════

NEW_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True
}

model = GPTModel(NEW_CONFIG)
print("Model created with random weights")

# ═══════════════════════════════════════════════════════════════
# Step 3: Load pretrained weights
# ═══════════════════════════════════════════════════════════════

load_weights_into_gpt(model, params)
print("✅ Pretrained weights loaded successfully!")

# ═══════════════════════════════════════════════════════════════
# Step 4: Verify weights are loaded
# ═══════════════════════════════════════════════════════════════

# Check token embeddings
print("\nToken embedding (first token, first 5 dims):")
print(model.tok_emb.weight[0, :5])

# Check if output head shares weights with token embeddings
print("\nWeight tying check:")
print(torch.equal(model.tok_emb.weight, model.out_head.weight))
# Should print: True

# ═══════════════════════════════════════════════════════════════
# Step 5: Test text generation
# ═══════════════════════════════════════════════════════════════

import tiktoken

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    return torch.tensor(encoded).unsqueeze(0)

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

# Generate text
tokenizer = tiktoken.get_encoding("gpt2")
model.eval()

start_text = "Hello, I am"
input_ids = text_to_token_ids(start_text, tokenizer)

output_ids = generate_text_simple(
    model=model,
    idx=input_ids,
    max_new_tokens=20,
    context_size=1024
)

generated_text = token_ids_to_text(output_ids, tokenizer)
print("\nGenerated text:")
print(generated_text)
# Should generate coherent text like:
# "Hello, I am a big fan of the show and I have been watching it for..."
```

---

## Visual Summary: Complete Weight Loading Flow

```
┌──────────────────────────────────────────────────────────────────┐
│ OpenAI's Pretrained Weights (params dictionary)                 │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│
│ • wte: Token embeddings (50257, 768)                            │
│ • wpe: Position embeddings (1024, 768)                          │
│ • blocks[0-11]: 12 transformer layers                           │
│   ├─ attn: Q,K,V weights + output projection                    │
│   ├─ mlp: Feed-forward weights (expansion + contraction)        │
│   └─ ln_1, ln_2: LayerNorm parameters                          │
│ • g, b: Final LayerNorm parameters                              │
└──────────────────────────────────────────────────────────────────┘
                              ↓
                   load_weights_into_gpt()
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ Your GPTModel (gpt)                                              │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│
│ ✅ tok_emb.weight ← params["wte"]                               │
│ ✅ pos_emb.weight ← params["wpe"]                               │
│ ✅ transformer_blocks[0-11]: All 12 layers loaded               │
│   ├─ attn.W_query/W_key/W_value ← Split Q,K,V weights          │
│   ├─ attn.out_proj ← Output projection                         │
│   ├─ ff.layers[0] ← Expansion layer                            │
│   ├─ ff.layers[2] ← Contraction layer                          │
│   └─ norm1, norm2 ← LayerNorm parameters                       │
│ ✅ final_norm ← Final LayerNorm                                 │
│ ✅ out_head.weight ← params["wte"] (weight tying)               │
└──────────────────────────────────────────────────────────────────┘
                              ↓
                    Ready to generate text! 🎉
```

---

## Key Takeaways

### ✅ What `load_weights_into_gpt()` does:

1. **Loads embeddings**: Token and positional embeddings (38.6M + 0.8M params)
2. **Processes 12 transformer blocks**: Attention, feed-forward, layer norms (85M params)
3. **Loads final normalization**: Scale and shift parameters (1.5K params)
4. **Implements weight tying**: Reuses embeddings for output (saves 38.6M params)

### 📋 Critical operations:

- **`np.split()`**: Splits concatenated Q,K,V weights
- **`.T` (transpose)**: Converts TensorFlow → PyTorch weight ordering
- **`assign()`**: Validates shapes and wraps as trainable parameters
- **Layer indexing**: Correctly targets `layers[0]` and `layers[2]` (skip GELU)

### 🎯 Result:

After running this function, your model goes from **random weights** to **OpenAI's pretrained weights**, enabling it to generate coherent text without any additional training!

### 💡 Remember:

```python
# Before:
model = GPTModel(config)  # Random weights ❌

# After:
load_weights_into_gpt(model, params)  # Pretrained weights ✅
model.eval()  # Ready for text generation! 🚀
```

The `load_weights_into_gpt()` function is your bridge from training a model from scratch to using a state-of-the-art pretrained language model!
