# Understanding Multi-Head Attention - A Complete Guide

## Overview

Multi-Head Attention is like having **multiple experts** looking at the same sentence, where each expert focuses on different aspects of the relationships between words.

Think of it like a team of editors:
- **Editor 1** might focus on grammar relationships
- **Editor 2** might focus on semantic meaning
- **Editor 3** might focus on contextual dependencies

Each "head" learns different patterns, and we combine their insights!

---

## Visual Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT TOKENS                                 │
│              Shape: (batch, num_tokens, d_in)                       │
│                     Example: (2, 6, 3)                              │
└────────────────────────┬────────────────────────────────────────────┘
                         │
            ┌────────────┼────────────┐
            ▼            ▼            ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │ W_query  │  │  W_key   │  │ W_value  │
    │ (3 → 8)  │  │ (3 → 8)  │  │ (3 → 8)  │
    └────┬─────┘  └────┬─────┘  └────┬─────┘
         │             │             │
         ▼             ▼             ▼
    ┌─────────────────────────────────────┐
    │    Q, K, V: (2, 6, 8)              │
    └──────────────┬──────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────────┐
    │  SPLIT INTO HEADS                    │
    │  .view(b, num_tokens, num_heads, hd) │
    │  (2, 6, 8) → (2, 6, 2, 4)           │
    └──────────────┬───────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────────┐
    │  TRANSPOSE FOR PARALLEL PROCESSING   │
    │  .transpose(1, 2)                    │
    │  (2, 6, 2, 4) → (2, 2, 6, 4)        │
    └──────────────┬───────────────────────┘
                   │
         ┌─────────┴─────────┐
         │                   │
    HEAD 1 (4-dim)      HEAD 2 (4-dim)
         │                   │
         ▼                   ▼
    ┌─────────────────────────────────┐
    │   ATTENTION SCORES              │
    │   Q @ K.T                       │
    │   (2, 2, 6, 4) @ (2, 2, 4, 6)  │
    │   → (2, 2, 6, 6)               │
    └──────────────┬──────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────┐
    │   APPLY CAUSAL MASK              │
    │   mask_fill(-inf for future)    │
    │   (2, 2, 6, 6)                  │
    └──────────────┬───────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────┐
    │   SOFTMAX (Scaled)               │
    │   softmax(scores / √head_dim)    │
    │   (2, 2, 6, 6)                  │
    └──────────────┬───────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────┐
    │   DROPOUT                        │
    │   (2, 2, 6, 6)                  │
    └──────────────┬───────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────┐
    │   WEIGHTED SUM OF VALUES         │
    │   attn_weights @ V               │
    │   (2, 2, 6, 6) @ (2, 2, 6, 4)   │
    │   → (2, 2, 6, 4)                │
    └──────────────┬───────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────┐
    │   TRANSPOSE BACK                 │
    │   .transpose(1, 2)               │
    │   (2, 2, 6, 4) → (2, 6, 2, 4)   │
    └──────────────┬───────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────┐
    │   CONCATENATE HEADS              │
    │   .view(b, num_tokens, d_out)    │
    │   (2, 6, 2, 4) → (2, 6, 8)      │
    └──────────────┬───────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────┐
    │   OUTPUT PROJECTION              │
    │   out_proj (8 → 8)               │
    │   (2, 6, 8) → (2, 6, 8)         │
    └──────────────┬───────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────┐
    │   FINAL OUTPUT                   │
    │   (batch, num_tokens, d_out)     │
    │   (2, 6, 8)                      │
    └──────────────────────────────────┘
```

---

## The Complete Multi-Head Attention Class

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, qkv_bias=False, dropout=0.0):
        super().__init__()
        assert (d_out % num_heads) == 0, "d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
```

---

## Part 1: Initialization - Setting Up the Heads

### Key Parameters

```python
d_in = 3           # Input embedding dimension (features per token)
d_out = 8          # Output embedding dimension (total across all heads)
num_heads = 2      # Number of attention heads
head_dim = 4       # d_out // num_heads = 8 // 2 = 4
```

### Why Must d_out Be Divisible by num_heads?

```python
assert (d_out % num_heads) == 0
```

**Example:**
- ✅ `d_out=8, num_heads=2` → `head_dim=4` (Valid!)
- ✅ `d_out=12, num_heads=3` → `head_dim=4` (Valid!)
- ❌ `d_out=7, num_heads=2` → `head_dim=3.5` (Invalid! Can't split evenly)

Each head needs the **same dimension** to work properly.

### The Weight Matrices

```python
self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)  # 3 → 8
self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)  # 3 → 8
self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)  # 3 → 8
```

**Important:** We create **ONE** large projection for all heads, then split it later!

**Visual:**
```
Input (d_in=3) ──→ [W_query] ──→ Output (d_out=8)
                                    ↓
                          Split into 2 heads of 4 dimensions each
```

### The Output Projection

```python
self.out_proj = nn.Linear(d_out, d_out)  # 8 → 8
```

This combines the outputs from all heads into a final representation.

---

## Part 2: The Forward Pass - Step by Step

### Input Shape

```python
x.shape = (batch_size=2, num_tokens=6, d_in=3)
```

**Example:**
- 2 sentences
- 6 words per sentence
- Each word is represented by 3 features

### Step 1: Project to Query, Key, Value

```python
keys = self.W_key(x)       # (2, 6, 3) → (2, 6, 8)
queries = self.W_query(x)  # (2, 6, 3) → (2, 6, 8)
values = self.W_value(x)   # (2, 6, 3) → (2, 6, 8)
```

Each token now has an 8-dimensional representation.

#### 🎯 WHY IS THIS IMPORTANT?

**Purpose:** Transform input tokens into three different representations (Q, K, V) that serve different roles in attention.

**Why we need it:**
1. **Learnable transformations**: The linear layers learn optimal representations for computing attention
2. **Increased expressiveness**: Moving from d_in=3 to d_out=8 gives the model more dimensions to work with
3. **Separate roles**: 
   - **Queries**: "What am I looking for?"
   - **Keys**: "What do I offer?"
   - **Values**: "What information do I carry?"
4. **Shared weights**: Using ONE set of projections for all heads is more parameter-efficient than separate projections per head

**Without this step:** We couldn't compute meaningful attention relationships between tokens!

### Step 2: Split into Multiple Heads

```python
# Reshape to add the num_heads dimension
keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
# (2, 6, 8) → (2, 6, 2, 4)
```

**Visual Breakdown:**

Before:
```
Token 1: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  (8 dimensions)
```

After splitting into 2 heads:
```
Token 1, Head 1: [0.1, 0.2, 0.3, 0.4]  (4 dimensions)
Token 1, Head 2: [0.5, 0.6, 0.7, 0.8]  (4 dimensions)
```

**Shape transformation:**
```
(batch, tokens, d_out) → (batch, tokens, num_heads, head_dim)
(2, 6, 8) → (2, 6, 2, 4)
```

#### 🎯 WHY IS THIS IMPORTANT?

**Purpose:** Partition the embedding space so different heads can learn different attention patterns.

**Why we need it:**
1. **Multiple perspectives**: Each head gets its own subspace (4 dimensions) to focus on different aspects
2. **Specialization**: Head 1 might learn syntactic patterns, Head 2 might learn semantic patterns
3. **No extra parameters**: We're just reorganizing existing dimensions, not creating new ones
4. **Parallel processing**: Setting up the structure so each head can work independently

**Real-world analogy:**
- Instead of having one 8-dimensional "super-expert"
- We have two 4-dimensional "specialized experts"
- Like having a grammar checker AND a style checker instead of just one general checker

**Without this step:** All attention would be computed in a single representation space, limiting the model's ability to capture diverse relationships!

### Step 3: Transpose for Efficient Computation

```python
keys = keys.transpose(1, 2)
queries = queries.transpose(1, 2)
values = values.transpose(1, 2)
# (2, 6, 2, 4) → (2, 2, 6, 4)
```

**Why transpose?**

We want each head to process all tokens independently:

**Before transpose:** `(batch, tokens, heads, head_dim)`
**After transpose:** `(batch, heads, tokens, head_dim)`

Now each head can work on the entire sequence independently!

**Visual:**
```
Before: Batch → Tokens → Heads → Features
After:  Batch → Heads → Tokens → Features
```

#### 🎯 WHY IS THIS IMPORTANT?

**Purpose:** Reorganize dimensions for efficient parallel matrix operations.

**Why we need it:**
1. **Parallel computation**: All heads can be processed simultaneously using batch matrix multiplication
2. **Memory efficiency**: GPU can process all heads in one operation instead of looping
3. **Proper broadcasting**: Sets up dimensions correctly for the upcoming `Q @ K.T` operation
4. **Computational speed**: Can be 10-100x faster than processing heads sequentially

**Technical benefit:**
```python
# Without transpose: Need to loop through heads (SLOW)
for head in range(num_heads):
    head_output = process_head(Q[:, :, head, :], K[:, :, head, :])

# With transpose: One operation for all heads (FAST)
all_heads_output = Q @ K.transpose(-2, -1)  # Processes all heads at once!
```

**Without this step:** We'd have to process each head one at a time, making training prohibitively slow for large models!

### Step 4: Compute Attention Scores

```python
attn_scores = queries @ keys.transpose(2, 3)
# (2, 2, 6, 4) @ (2, 2, 4, 6) → (2, 2, 6, 6)
```

**Shape breakdown:**
- `(batch=2, num_heads=2, num_tokens=6, head_dim=4)`
- `@` with transpose of keys: `(batch=2, num_heads=2, head_dim=4, num_tokens=6)`
- **Result:** `(batch=2, num_heads=2, num_tokens=6, num_tokens=6)`

**What this means:**
- For **each batch**
- For **each head**
- We have a **6×6 matrix** of attention scores (token-to-token relationships)

**Example for one head:**
```
        Your  journey  starts  with  one  step
Your    [0.9,   0.1,    0.0,   0.0,  0.0, 0.0]
journey [0.3,   0.6,    0.1,   0.0,  0.0, 0.0]
starts  [0.2,   0.2,    0.5,   0.1,  0.0, 0.0]
with    [0.1,   0.1,    0.3,   0.4,  0.1, 0.0]
one     [0.1,   0.1,    0.2,   0.2,  0.3, 0.1]
step    [0.1,   0.1,    0.1,   0.2,  0.2, 0.3]
```

#### 🎯 WHY IS THIS IMPORTANT?

**Purpose:** Calculate compatibility scores between every pair of tokens - this is the CORE of attention!

**Why we need it:**
1. **Relationship measurement**: The dot product measures how "similar" or "compatible" each query is with each key
2. **High score = strong relationship**: If Query(token A) ⋅ Key(token B) is high, token A should pay attention to token B
3. **All pairwise comparisons**: The 6×6 matrix contains scores for ALL possible token relationships
4. **Different patterns per head**: Each head computes its own scores, finding different types of relationships

**Mathematical meaning:**
```
score[i, j] = Query[i] · Key[j]

High score → "Token i finds token j relevant"
Low score  → "Token i doesn't need information from token j"
```

**Concrete example:**
```
Query("bank") · Key("river") = 0.8  (high - related in geography context)
Query("bank") · Key("money") = 0.9  (high - related in finance context)
Query("bank") · Key("hello") = 0.1  (low - not related)
```

**Without this step:** We'd have no way to determine which tokens should attend to which other tokens!

### Step 5: Apply Causal Mask

```python
mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
attn_scores.masked_fill_(mask_bool, -torch.inf)
```

**Before masking:**
```
[[ 0.5,  0.3,  0.2,  0.1,  0.4,  0.2],
 [ 0.2,  0.6,  0.3,  0.1,  0.2,  0.3],
 [ 0.1,  0.2,  0.7,  0.2,  0.1,  0.1],
 [ 0.3,  0.1,  0.2,  0.5,  0.3,  0.2],
 [ 0.2,  0.2,  0.1,  0.3,  0.6,  0.4],
 [ 0.1,  0.3,  0.2,  0.2,  0.3,  0.7]]
```

**After masking (future tokens = -inf):**
```
[[ 0.5,  -inf,  -inf,  -inf,  -inf,  -inf],
 [ 0.2,  0.6,  -inf,  -inf,  -inf,  -inf],
 [ 0.1,  0.2,  0.7,  -inf,  -inf,  -inf],
 [ 0.3,  0.1,  0.2,  0.5,  -inf,  -inf],
 [ 0.2,  0.2,  0.1,  0.3,  0.6,  -inf],
 [ 0.1,  0.3,  0.2,  0.2,  0.3,  0.7]]
```

This ensures tokens can only attend to previous tokens (causal attention).

#### 🎯 WHY IS THIS IMPORTANT?

**Purpose:** Prevent tokens from "seeing into the future" - crucial for autoregressive language models!

**Why we need it:**
1. **Maintain causality**: Token at position t can only use information from positions ≤ t
2. **Training = Testing**: During training, we mask future tokens to match what happens during generation
3. **No information leakage**: Without masking, the model could "cheat" by looking at future words
4. **Autoregressive generation**: Required for models that generate text one token at a time

**Real-world example:**
```
Sentence: "The cat sat on the"

When predicting "mat":
✅ Can see: "The cat sat on the" (past context)
❌ Cannot see: future tokens that haven't been generated yet

Without masking, during training the model could see "mat" while predicting "on", 
making it impossible to generate properly during inference!
```

**Why -inf specifically?**
- After softmax, `e^(-inf) = 0`
- Future tokens get exactly 0% attention weight
- Clean and mathematically elegant solution

**Without this step:** The model would learn patterns that don't work during actual text generation!

### Step 6: Apply Softmax and Dropout

```python
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
attn_weights = self.dropout(attn_weights)
```

**After softmax:**
```
[[1.0,  0.0,  0.0,  0.0,  0.0,  0.0],
 [0.4,  0.6,  0.0,  0.0,  0.0,  0.0],
 [0.2,  0.3,  0.5,  0.0,  0.0,  0.0],
 [0.3,  0.1,  0.2,  0.4,  0.0,  0.0],
 [0.1,  0.2,  0.1,  0.2,  0.4,  0.0],
 [0.1,  0.2,  0.1,  0.1,  0.2,  0.3]]
```

Each row now sums to 1.0!

#### 🎯 WHY IS THIS IMPORTANT?

**Purpose:** Convert attention scores into probability distributions and add regularization.

**Why we need it:**

**1. Softmax normalization:**
- Converts arbitrary scores into probabilities (0 to 1, sum to 1)
- Makes interpretation clear: "Token A gives 40% attention to Token B"
- Ensures numerical stability for the weighted sum

**2. Scaling by √head_dim:**
- Prevents attention scores from becoming too large as dimensions increase
- Keeps gradient magnitudes stable during training
- Without scaling: attention becomes too "peaked" (almost one-hot)
- Review the scaling explanation document for full mathematical justification!

**3. Dropout regularization:**
- Randomly zeros out some attention weights during training
- Forces the model to not rely too heavily on specific attention patterns
- Prevents overfitting by making the model more robust
- During inference (eval mode), dropout is disabled

**Concrete example:**
```
Before softmax: [-2.3, 0.5, 1.2, -inf, -inf, -inf]
After softmax:  [0.02, 0.37, 0.61, 0.0, 0.0, 0.0]  (probabilities!)

After dropout (50%): [0.0, 0.74, 1.22, 0.0, 0.0, 0.0]  (some zeroed, others scaled)
```

**Without this step:** 
- No probabilistic interpretation
- Unstable training for large dimensions
- Increased overfitting risk

### Step 7: Apply Attention to Values

```python
context_vec = (attn_weights @ values).transpose(1, 2)
# (2, 2, 6, 6) @ (2, 2, 6, 4) → (2, 2, 6, 4)
# Then transpose: (2, 2, 6, 4) → (2, 6, 2, 4)
```

This computes the weighted sum of values for each token, for each head.

#### 🎯 WHY IS THIS IMPORTANT?

**Purpose:** Create the final contextualized representation by mixing information from relevant tokens!

**Why we need it:**
1. **Information aggregation**: Combine information from all attended tokens based on attention weights
2. **Context-aware representations**: Each token's output now contains information from other tokens it paid attention to
3. **Weighted mixture**: Higher attention weight = more influence from that token's value

**Mathematical interpretation:**
```
context[i] = Σ(attention_weight[i,j] × value[j])

For token "bank" in "river bank":
context("bank") = 0.1×value("river") + 0.7×value("bank") + 0.2×value("flows")
                = mixture emphasizing its own meaning + river context
```

**Concrete example:**
```
Token: "bank"
Attention weights: [0.1 (river), 0.7 (bank), 0.2 (flows)]
Values: 
  river: [0.2, 0.8, 0.1, 0.5]
  bank:  [0.5, 0.3, 0.6, 0.2]
  flows: [0.1, 0.7, 0.2, 0.4]

Context("bank") = 0.1×[0.2,0.8,0.1,0.5] + 0.7×[0.5,0.3,0.6,0.2] + 0.2×[0.1,0.7,0.2,0.4]
                = [0.39, 0.51, 0.47, 0.28]  (mixed representation!)
```

**Why transpose back?**
- Changes from `(batch, heads, tokens, head_dim)` to `(batch, tokens, heads, head_dim)`
- Prepares for concatenating heads in the next step

**Without this step:** We'd have attention weights but no way to actually use them to create enriched representations!

### Step 8: Combine Heads

```python
context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
# (2, 6, 2, 4) → (2, 6, 8)
```

**What's happening:**

Before:
```
Token 1, Head 1: [0.1, 0.2, 0.3, 0.4]
Token 1, Head 2: [0.5, 0.6, 0.7, 0.8]
```

After concatenation:
```
Token 1: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  (8 dimensions)
```

#### 🎯 WHY IS THIS IMPORTANT?

**Purpose:** Merge the different perspectives from all heads into a single unified representation.

**Why we need it:**
1. **Multi-perspective fusion**: Combines insights from all heads (grammar + semantics + context + ...)
2. **Restore original shape**: Returns to `(batch, tokens, d_out)` matching the expected output shape
3. **Richer representation**: Each position now has 8 dimensions containing diverse information from 2 different 4-dim perspectives

**What `.contiguous()` does:**
- Ensures memory layout is contiguous before reshaping
- PyTorch requirement when using `.view()` after `.transpose()`
- Prevents memory errors during the reshape operation

**Analogy:**
```
Think of it as combining reports from multiple experts:

Head 1 report (grammar): "Subject-verb agreement is correct"
Head 2 report (semantics): "Meaning is about financial institution"

Combined report: "Grammatically correct sentence about a financial institution"
```

**Real tensor example:**
```
Head 1 focused on syntax: [0.2, 0.3, 0.1, 0.5]  (learned syntactic features)
Head 2 focused on semantics: [0.7, 0.1, 0.8, 0.2]  (learned semantic features)

Concatenated: [0.2, 0.3, 0.1, 0.5, 0.7, 0.1, 0.8, 0.2]  (both perspectives!)
```

**Without this step:** We'd have separate head outputs with no way to use them together!

### Step 9: Final Projection

```python
context_vec = self.out_proj(context_vec)
# (2, 6, 8) → (2, 6, 8)
```

This optional projection allows the model to learn how to best combine the head outputs.

#### 🎯 WHY IS THIS IMPORTANT?

**Purpose:** Learn the optimal way to integrate information from all heads through a trainable transformation.

**Why we need it:**
1. **Learnable fusion**: Instead of simple concatenation, the model learns HOW to best combine head outputs
2. **Dimension flexibility**: Can change output dimension if needed (though here it's 8→8)
3. **Additional expressiveness**: Adds another layer of non-linearity and transformation
4. **Standard in transformers**: Used in original "Attention is All You Need" paper and all modern implementations

**What it's learning:**
```python
# The out_proj learns a weight matrix W_O of shape (8, 8)

# It can learn patterns like:
# "When head 1 says X and head 2 says Y, the final output should emphasize Z"

# Example learned behavior:
If head1[0:4] indicates subject && head2[4:8] indicates verb:
    emphasize agreement features in output
```

**Concrete example:**
```
Input (concatenated heads): [0.2, 0.3, 0.1, 0.5, 0.7, 0.1, 0.8, 0.2]

After out_proj transformation:
Output: [0.4, 0.5, 0.3, 0.6, 0.2, 0.7, 0.4, 0.3]

The network learned to reweight and recombine features optimally for the task!
```

**Benefits:**
- **Cross-head interactions**: Allows features from different heads to interact
- **Task-specific adaptation**: Different tasks may need different combinations of head outputs
- **Gradient flow**: Provides additional path for gradients during backpropagation

**Could we skip it?**
- Technically yes, but performance would degrade
- It's a relatively small addition (d_out × d_out parameters)
- The learned transformation is crucial for optimal performance

**Without this step:** The model loses the ability to learn optimal ways to combine multi-head information, reducing overall capacity!

---

## Complete Example with Real Numbers

### Setup

```python
batch_size = 2
num_tokens = 6
d_in = 3
d_out = 8
num_heads = 2
head_dim = 4  # d_out // num_heads

# Input: (2, 6, 3)
x = torch.randn(2, 6, 3)
```

### Shape Transformations Throughout

```
Input:              (2, 6, 3)
    ↓ W_query/key/value
Q, K, V:            (2, 6, 8)
    ↓ view (split heads)
Q, K, V:            (2, 6, 2, 4)
    ↓ transpose
Q, K, V:            (2, 2, 6, 4)
    ↓ attention scores
Scores:             (2, 2, 6, 6)
    ↓ softmax
Weights:            (2, 2, 6, 6)
    ↓ @ values
Context:            (2, 2, 6, 4)
    ↓ transpose
Context:            (2, 6, 2, 4)
    ↓ view (combine heads)
Context:            (2, 6, 8)
    ↓ out_proj
Output:             (2, 6, 8)
```

---

## Key Differences: MultiHeadAttentionWrapper vs MultiHeadAttention

### MultiHeadAttentionWrapper (Simple but Inefficient)

```python
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # Creates SEPARATE CausalAttention for each head
        self.heads = nn.ModuleList([
            CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) 
            for _ in range(num_heads)
        ])
    
    def forward(self, x):
        # Process each head separately and concatenate
        return torch.cat([head(x) for head in self.heads], dim=-1)
```

**Problems:**
- Each head has its own separate weight matrices (more parameters)
- Processes heads sequentially (slower)
- Output dimension = `num_heads * d_out`

### MultiHeadAttention (Efficient)

**Advantages:**
- Single set of weight matrices, split internally (fewer parameters)
- Processes all heads in parallel (faster)
- Output dimension = `d_out` (more flexible)
- Standard implementation used in transformers

---

## Why Multiple Heads?

### Analogy: Looking at a Painting

Imagine you're analyzing a painting:

**Single Head (Single Attention):**
- You can only focus on one aspect at a time
- Maybe you notice the colors OR the shapes OR the emotions

**Multiple Heads (Multi-Head Attention):**
- **Head 1:** Focuses on colors and lighting
- **Head 2:** Focuses on shapes and composition
- **Head 3:** Focuses on emotions and subject matter
- **Head 4:** Focuses on brushstroke technique

By combining all perspectives, you get a **richer understanding** of the painting!

### In Language Models

**Example sentence:** "The bank can guarantee deposits will eventually cover future tuition costs."

**Different heads might focus on:**
- **Head 1:** Financial relationships (bank → deposits → costs)
- **Head 2:** Temporal relationships (eventually → future)
- **Head 3:** Modal relationships (can → will)
- **Head 4:** Semantic relationships (guarantee → cover)

---

## Summary: The Big Picture

1. **Single weight matrices** project to `d_out` dimensions
2. **Split** the `d_out` dimensions across `num_heads` heads
3. **Each head** learns different attention patterns independently
4. **Parallel processing** makes it efficient
5. **Combine** all head outputs
6. **Final projection** integrates the multi-perspective information

**Formula:**
```
MultiHead(Q, K, V) = Concat(head₁, head₂, ..., headₙ) W^O

where headᵢ = Attention(QWᵢQ, KWᵢK, VWᵢV)
```

This architecture is the **foundation of transformers** and models like GPT, BERT, and all modern LLMs! 🚀

---

---

---

# 📐 Complete Mathematical Example: Multi-Head Attention Step-by-Step

## Setup

Let's work through a **complete example with actual numbers** to see exactly how multi-head attention works!

**Parameters:**
- **num_tokens = 4** (4 words in our sentence)
- **d_in = 3** (each token has 3 input features)
- **d_out = 2** (output will have 2 dimensions total)
- **num_heads = 2** (we'll split into 2 attention heads)
- **head_dim = 1** (d_out / num_heads = 2 / 2 = 1 dimension per head)

**Input Sentence:** "The cat sat down"

---

## Step 0: Input Representation

Our input tokens are embedded in 3-dimensional space:

```
X = [x₁]   [0.5]
    [x₂] = [0.3]  ← Token 1: "The"
    [x₃]   [0.8]

    [x₁]   [0.6]
    [x₂] = [0.9]  ← Token 2: "cat"
    [x₃]   [0.4]

    [x₁]   [0.2]
    [x₂] = [0.7]  ← Token 3: "sat"
    [x₃]   [0.5]

    [x₁]   [0.8]
    [x₂] = [0.1]  ← Token 4: "down"
    [x₃]   [0.6]
```

**Matrix form (4 tokens × 3 features):**
```
     ┌          ┐
     │ 0.5 0.3 0.8 │  ← Token 1
X =  │ 0.6 0.9 0.4 │  ← Token 2
     │ 0.2 0.7 0.5 │  ← Token 3
     │ 0.8 0.1 0.6 │  ← Token 4
     └          ┘
```

---

## Step 1: Project to Query, Key, Value

We need three weight matrices to project from d_in=3 to d_out=2.

### Initialize Weight Matrices (simplified for clarity)

**W_query (3 × 2):**
```
     ┌        ┐
     │ 0.4  0.2 │
Wq = │ 0.1  0.3 │
     │ 0.5  0.1 │
     └        ┘
```

**W_key (3 × 2):**
```
     ┌        ┐
     │ 0.3  0.4 │
Wk = │ 0.2  0.1 │
     │ 0.1  0.5 │
     └        ┘
```

**W_value (3 × 2):**
```
     ┌        ┐
     │ 0.6  0.1 │
Wv = │ 0.2  0.4 │
     │ 0.3  0.2 │
     └        ┘
```

### Compute Queries (Q = X × W_query)

**For Token 1 "The": [0.5, 0.3, 0.8]**
```
q₁ = [0.5  0.3  0.8] × [0.4  0.2]   = [0.5×0.4 + 0.3×0.1 + 0.8×0.5,  0.5×0.2 + 0.3×0.3 + 0.8×0.1]
                        [0.1  0.3]
                        [0.5  0.1]
   = [0.20 + 0.03 + 0.40,  0.10 + 0.09 + 0.08]
   = [0.63, 0.27]
```

**For Token 2 "cat": [0.6, 0.9, 0.4]**
```
q₂ = [0.6  0.9  0.4] × Wq = [0.6×0.4 + 0.9×0.1 + 0.4×0.5,  0.6×0.2 + 0.9×0.3 + 0.4×0.1]
   = [0.24 + 0.09 + 0.20,  0.12 + 0.27 + 0.04]
   = [0.53, 0.43]
```

**For Token 3 "sat": [0.2, 0.7, 0.5]**
```
q₃ = [0.2  0.7  0.5] × Wq = [0.2×0.4 + 0.7×0.1 + 0.5×0.5,  0.2×0.2 + 0.7×0.3 + 0.5×0.1]
   = [0.08 + 0.07 + 0.25,  0.04 + 0.21 + 0.05]
   = [0.40, 0.30]
```

**For Token 4 "down": [0.8, 0.1, 0.6]**
```
q₄ = [0.8  0.1  0.6] × Wq = [0.8×0.4 + 0.1×0.1 + 0.6×0.5,  0.8×0.2 + 0.1×0.3 + 0.6×0.1]
   = [0.32 + 0.01 + 0.30,  0.16 + 0.03 + 0.06]
   = [0.63, 0.25]
```

**Queries Matrix Q (4 × 2):**
```
    ┌          ┐
    │ 0.63  0.27 │  ← q₁
Q = │ 0.53  0.43 │  ← q₂
    │ 0.40  0.30 │  ← q₃
    │ 0.63  0.25 │  ← q₄
    └          ┘
```

### Compute Keys (K = X × W_key)

**For Token 1 "The":**
```
k₁ = [0.5  0.3  0.8] × [0.3  0.4]   = [0.5×0.3 + 0.3×0.2 + 0.8×0.1,  0.5×0.4 + 0.3×0.1 + 0.8×0.5]
                        [0.2  0.1]
                        [0.1  0.5]
   = [0.15 + 0.06 + 0.08,  0.20 + 0.03 + 0.40]
   = [0.29, 0.63]
```

**For Token 2 "cat":**
```
k₂ = [0.6  0.9  0.4] × Wk = [0.6×0.3 + 0.9×0.2 + 0.4×0.1,  0.6×0.4 + 0.9×0.1 + 0.4×0.5]
   = [0.18 + 0.18 + 0.04,  0.24 + 0.09 + 0.20]
   = [0.40, 0.53]
```

**For Token 3 "sat":**
```
k₃ = [0.2  0.7  0.5] × Wk = [0.2×0.3 + 0.7×0.2 + 0.5×0.1,  0.2×0.4 + 0.7×0.1 + 0.5×0.5]
   = [0.06 + 0.14 + 0.05,  0.08 + 0.07 + 0.25]
   = [0.25, 0.40]
```

**For Token 4 "down":**
```
k₄ = [0.8  0.1  0.6] × Wk = [0.8×0.3 + 0.1×0.2 + 0.6×0.1,  0.8×0.4 + 0.1×0.1 + 0.6×0.5]
   = [0.24 + 0.02 + 0.06,  0.32 + 0.01 + 0.30]
   = [0.32, 0.63]
```

**Keys Matrix K (4 × 2):**
```
    ┌          ┐
    │ 0.29  0.63 │  ← k₁
K = │ 0.40  0.53 │  ← k₂
    │ 0.25  0.40 │  ← k₃
    │ 0.32  0.63 │  ← k₄
    └          ┘
```

### Compute Values (V = X × W_value)

**For Token 1 "The":**
```
v₁ = [0.5  0.3  0.8] × [0.6  0.1]   = [0.5×0.6 + 0.3×0.2 + 0.8×0.3,  0.5×0.1 + 0.3×0.4 + 0.8×0.2]
                        [0.2  0.4]
                        [0.3  0.2]
   = [0.30 + 0.06 + 0.24,  0.05 + 0.12 + 0.16]
   = [0.60, 0.33]
```

**For Token 2 "cat":**
```
v₂ = [0.6  0.9  0.4] × Wv = [0.6×0.6 + 0.9×0.2 + 0.4×0.3,  0.6×0.1 + 0.9×0.4 + 0.4×0.2]
   = [0.36 + 0.18 + 0.12,  0.06 + 0.36 + 0.08]
   = [0.66, 0.50]
```

**For Token 3 "sat":**
```
v₃ = [0.2  0.7  0.5] × Wv = [0.2×0.6 + 0.7×0.2 + 0.5×0.3,  0.2×0.1 + 0.7×0.4 + 0.5×0.2]
   = [0.12 + 0.14 + 0.15,  0.02 + 0.28 + 0.10]
   = [0.41, 0.40]
```

**For Token 4 "down":**
```
v₄ = [0.8  0.1  0.6] × Wv = [0.8×0.6 + 0.1×0.2 + 0.6×0.3,  0.8×0.1 + 0.1×0.4 + 0.6×0.2]
   = [0.48 + 0.02 + 0.18,  0.08 + 0.04 + 0.12]
   = [0.68, 0.24]
```

**Values Matrix V (4 × 2):**
```
    ┌          ┐
    │ 0.60  0.33 │  ← v₁
V = │ 0.66  0.50 │  ← v₂
    │ 0.41  0.40 │  ← v₃
    │ 0.68  0.24 │  ← v₄
    └          ┘
```

---

## Step 2: Split into Multiple Heads

Now we split Q, K, V into 2 heads. Since d_out=2 and num_heads=2, each head gets head_dim=1.

**Queries split into heads:**
```
Q_head1 = [0.63]  (column 1)    Q_head2 = [0.27]  (column 2)
          [0.53]                          [0.43]
          [0.40]                          [0.30]
          [0.63]                          [0.25]
```

**Keys split into heads:**
```
K_head1 = [0.29]  (column 1)    K_head2 = [0.63]  (column 2)
          [0.40]                          [0.53]
          [0.25]                          [0.40]
          [0.32]                          [0.63]
```

**Values split into heads:**
```
V_head1 = [0.60]  (column 1)    V_head2 = [0.33]  (column 2)
          [0.66]                          [0.50]
          [0.41]                          [0.40]
          [0.68]                          [0.24]
```

---

## Step 3: Compute Attention for Head 1

### 3a. Compute Attention Scores (Q · K^T)

```
Scores_head1 = Q_head1 × K_head1^T

K_head1^T = [0.29  0.40  0.25  0.32]  (1 × 4)

For each query:
score₁₁ = 0.63 × 0.29 = 0.183
score₁₂ = 0.63 × 0.40 = 0.252
score₁₃ = 0.63 × 0.25 = 0.158
score₁₄ = 0.63 × 0.32 = 0.202

score₂₁ = 0.53 × 0.29 = 0.154
score₂₂ = 0.53 × 0.40 = 0.212
score₂₃ = 0.53 × 0.25 = 0.133
score₂₄ = 0.53 × 0.32 = 0.170

score₃₁ = 0.40 × 0.29 = 0.116
score₃₂ = 0.40 × 0.40 = 0.160
score₃₃ = 0.40 × 0.25 = 0.100
score₃₄ = 0.40 × 0.32 = 0.128

score₄₁ = 0.63 × 0.29 = 0.183
score₄₂ = 0.63 × 0.40 = 0.252
score₄₃ = 0.63 × 0.25 = 0.158
score₄₄ = 0.63 × 0.32 = 0.202
```

**Attention Scores Matrix (4 × 4):**
```
                Token1  Token2  Token3  Token4
         ┌                                    ┐
Token1   │  0.183  0.252  0.158  0.202 │
Token2   │  0.154  0.212  0.133  0.170 │
Token3   │  0.116  0.160  0.100  0.128 │
Token4   │  0.183  0.252  0.158  0.202 │
         └                                    ┘
```

### 3b. Apply Causal Mask

Replace future positions with -∞:

```
                Token1  Token2  Token3  Token4
         ┌                                    ┐
Token1   │  0.183   -∞      -∞      -∞   │  (can only see token 1)
Token2   │  0.154  0.212    -∞      -∞   │  (can see tokens 1-2)
Token3   │  0.116  0.160  0.100    -∞   │  (can see tokens 1-3)
Token4   │  0.183  0.252  0.158  0.202 │  (can see all tokens)
         └                                    ┘
```

### 3c. Scale and Apply Softmax

**Scaling factor:** √(head_dim) = √1 = 1 (so no actual scaling needed here!)

**Apply softmax to each row:**

**Row 1 (Token 1 "The"):**
```
Only one value: 0.183
After softmax: e^0.183 / e^0.183 = 1.0

Weights: [1.0, 0, 0, 0]
```

**Row 2 (Token 2 "cat"):**
```
Values: [0.154, 0.212]
e^0.154 = 1.167
e^0.212 = 1.236
Sum = 1.167 + 1.236 = 2.403

Weights: [1.167/2.403, 1.236/2.403, 0, 0]
       = [0.486, 0.514, 0, 0]
```

**Row 3 (Token 3 "sat"):**
```
Values: [0.116, 0.160, 0.100]
e^0.116 = 1.123
e^0.160 = 1.174
e^0.100 = 1.105
Sum = 1.123 + 1.174 + 1.105 = 3.402

Weights: [1.123/3.402, 1.174/3.402, 1.105/3.402, 0]
       = [0.330, 0.345, 0.325, 0]
```

**Row 4 (Token 4 "down"):**
```
Values: [0.183, 0.252, 0.158, 0.202]
e^0.183 = 1.201
e^0.252 = 1.287
e^0.158 = 1.171
e^0.202 = 1.224
Sum = 1.201 + 1.287 + 1.171 + 1.224 = 4.883

Weights: [1.201/4.883, 1.287/4.883, 1.171/4.883, 1.224/4.883]
       = [0.246, 0.264, 0.240, 0.251]
```

**Attention Weights Matrix (Head 1):**
```
         ┌                                ┐
         │  1.000  0.000  0.000  0.000 │
W_head1= │  0.486  0.514  0.000  0.000 │
         │  0.330  0.345  0.325  0.000 │
         │  0.246  0.264  0.240  0.251 │
         └                                ┘
```

### 3d. Apply Attention to Values

```
Context_head1 = W_head1 × V_head1

For Token 1:
c₁ = 1.000×0.60 + 0.000×0.66 + 0.000×0.41 + 0.000×0.68
   = 0.60

For Token 2:
c₂ = 0.486×0.60 + 0.514×0.66 + 0.000×0.41 + 0.000×0.68
   = 0.292 + 0.339 = 0.631

For Token 3:
c₃ = 0.330×0.60 + 0.345×0.66 + 0.325×0.41 + 0.000×0.68
   = 0.198 + 0.228 + 0.133 = 0.559

For Token 4:
c₄ = 0.246×0.60 + 0.264×0.66 + 0.240×0.41 + 0.251×0.68
   = 0.148 + 0.174 + 0.098 + 0.171 = 0.591
```

**Context from Head 1:**
```
Context_head1 = [0.600]
                [0.631]
                [0.559]
                [0.591]
```

---

## Step 4: Compute Attention for Head 2

Following the same process for Head 2...

### 4a. Attention Scores

```
Scores_head2 = Q_head2 × K_head2^T

K_head2^T = [0.63  0.53  0.40  0.63]

score₁₁ = 0.27 × 0.63 = 0.170
score₁₂ = 0.27 × 0.53 = 0.143
score₁₃ = 0.27 × 0.40 = 0.108
score₁₄ = 0.27 × 0.63 = 0.170

score₂₁ = 0.43 × 0.63 = 0.271
score₂₂ = 0.43 × 0.53 = 0.228
score₂₃ = 0.43 × 0.40 = 0.172
score₂₄ = 0.43 × 0.63 = 0.271

score₃₁ = 0.30 × 0.63 = 0.189
score₃₂ = 0.30 × 0.53 = 0.159
score₃₃ = 0.30 × 0.40 = 0.120
score₃₄ = 0.30 × 0.63 = 0.189

score₄₁ = 0.25 × 0.63 = 0.158
score₄₂ = 0.25 × 0.53 = 0.133
score₄₃ = 0.25 × 0.40 = 0.100
score₄₄ = 0.25 × 0.63 = 0.158
```

### 4b. Apply Causal Mask and Softmax

**After masking:**
```
         ┌                                ┐
         │  0.170   -∞      -∞      -∞   │
         │  0.271  0.228    -∞      -∞   │
         │  0.189  0.159  0.120    -∞   │
         │  0.158  0.133  0.100  0.158 │
         └                                ┘
```

**After softmax:**

Row 1: [1.0, 0, 0, 0]

Row 2:
```
e^0.271 = 1.311,  e^0.228 = 1.256
Sum = 2.567
Weights: [0.511, 0.489, 0, 0]
```

Row 3:
```
e^0.189 = 1.208,  e^0.159 = 1.172,  e^0.120 = 1.127
Sum = 3.507
Weights: [0.344, 0.334, 0.322, 0]
```

Row 4:
```
e^0.158 = 1.171,  e^0.133 = 1.142,  e^0.100 = 1.105,  e^0.158 = 1.171
Sum = 4.589
Weights: [0.255, 0.249, 0.241, 0.255]
```

**Attention Weights Matrix (Head 2):**
```
         ┌                                ┐
         │  1.000  0.000  0.000  0.000 │
W_head2= │  0.511  0.489  0.000  0.000 │
         │  0.344  0.334  0.322  0.000 │
         │  0.255  0.249  0.241  0.255 │
         └                                ┘
```

### 4c. Apply Attention to Values

```
Context_head2 = W_head2 × V_head2

For Token 1:
c₁ = 1.000×0.33 = 0.330

For Token 2:
c₂ = 0.511×0.33 + 0.489×0.50
   = 0.169 + 0.245 = 0.414

For Token 3:
c₃ = 0.344×0.33 + 0.334×0.50 + 0.322×0.40
   = 0.114 + 0.167 + 0.129 = 0.410

For Token 4:
c₄ = 0.255×0.33 + 0.249×0.50 + 0.241×0.40 + 0.255×0.24
   = 0.084 + 0.125 + 0.096 + 0.061 = 0.366
```

**Context from Head 2:**
```
Context_head2 = [0.330]
                [0.414]
                [0.410]
                [0.366]
```

---

## Step 5: Concatenate Heads

Now we combine the outputs from both heads:

```
         Head1   Head2
Token1: [0.600, 0.330]
Token2: [0.631, 0.414]
Token3: [0.559, 0.410]
Token4: [0.591, 0.366]
```

**Combined Context Matrix (4 × 2):**
```
            ┌              ┐
            │ 0.600  0.330 │
Context =   │ 0.631  0.414 │
            │ 0.559  0.410 │
            │ 0.591  0.366 │
            └              ┘
```

---

## Step 6: Output Projection

Finally, we apply the output projection W_out (2 × 2):

**W_out:**
```
        ┌          ┐
        │ 0.7  0.3 │
W_out = │ 0.4  0.6 │
        └          ┘
```

**For Token 1 "The":**
```
output₁ = [0.600  0.330] × [0.7  0.3]
                            [0.4  0.6]
        = [0.600×0.7 + 0.330×0.4,  0.600×0.3 + 0.330×0.6]
        = [0.420 + 0.132,  0.180 + 0.198]
        = [0.552, 0.378]
```

**For Token 2 "cat":**
```
output₂ = [0.631  0.414] × W_out
        = [0.631×0.7 + 0.414×0.4,  0.631×0.3 + 0.414×0.6]
        = [0.442 + 0.166,  0.189 + 0.248]
        = [0.608, 0.437]
```

**For Token 3 "sat":**
```
output₃ = [0.559  0.410] × W_out
        = [0.559×0.7 + 0.410×0.4,  0.559×0.3 + 0.410×0.6]
        = [0.391 + 0.164,  0.168 + 0.246]
        = [0.555, 0.414]
```

**For Token 4 "down":**
```
output₄ = [0.591  0.366] × W_out
        = [0.591×0.7 + 0.366×0.4,  0.591×0.3 + 0.366×0.6]
        = [0.414 + 0.146,  0.177 + 0.220]
        = [0.560, 0.397]
```

---

## Final Output

**Multi-Head Attention Output (4 × 2):**
```
              ┌              ┐
              │ 0.552  0.378 │  ← Token 1 "The"
Final Output =│ 0.608  0.437 │  ← Token 2 "cat"
              │ 0.555  0.414 │  ← Token 3 "sat"
              │ 0.560  0.397 │  ← Token 4 "down"
              └              ┘
```

---

## Summary of What Happened

1. **Input:** Started with 4 tokens, each with 3 features
2. **Projection:** Transformed to Query, Key, Value using learned weight matrices
3. **Split Heads:** Divided d_out=2 into 2 heads of dimension 1 each
4. **Head 1 Attention:** Computed attention relationships and created context vectors
5. **Head 2 Attention:** Computed different attention relationships (different perspective)
6. **Concatenate:** Combined both heads' outputs back together
7. **Final Projection:** Transformed the combined output through W_out

**Key Insight:** Each head learned to focus on different relationships:
- Head 1 might have learned syntactic dependencies
- Head 2 might have learned semantic relationships

By combining both perspectives, we get richer, more informative token representations!

---

## Verification

Let's verify our output makes sense:

✅ **Input shape:** 4 tokens × 3 features
✅ **Output shape:** 4 tokens × 2 features
✅ **Causal masking:** Token 1 only sees itself, Token 2 sees tokens 1-2, etc.
✅ **Attention weights:** All rows sum to 1.0 (probability distributions)
✅ **Context vectors:** Each token's output is a weighted combination of value vectors
✅ **Multiple perspectives:** Two different attention patterns were learned and combined

This is **exactly** how multi-head attention works in transformers like GPT! 🎯
