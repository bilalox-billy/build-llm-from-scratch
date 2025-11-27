# Text Generation in GPT Models: A Complete Guide

## Overview

The `generate_text()` function is the heart of autoregressive text generation in GPT models. It takes a starting prompt and generates new tokens one at a time, appending each predicted token back to the context to predict the next one.

## Function Signature

```python
def generate_text(model, idx, max_new_tokens, context_size):
    """
    Generate text autoregressively using a trained GPT model.
    
    Parameters:
    -----------
    model : GPTModel
        The trained transformer model
    idx : torch.Tensor
        Input token indices, shape (batch_size, n_tokens)
    max_new_tokens : int
        Number of new tokens to generate
    context_size : int
        Maximum context length the model can handle (e.g., 1024 for GPT-2)
    
    Returns:
    --------
    torch.Tensor
        Extended sequence with generated tokens, shape (batch_size, n_tokens + max_new_tokens)
    """
```

---

## The Autoregressive Generation Process

### What is Autoregressive Generation?

**Autoregressive** means the model generates one token at a time, using previously generated tokens as context for predicting the next token.

```
Step 1: "Hello, I"          → predict "am"
Step 2: "Hello, I am"       → predict "a"
Step 3: "Hello, I am a"     → predict "student"
Step 4: "Hello, I am a student" → predict "."
```

Each prediction depends on **all previous tokens**, creating a sequential dependency.

---

## Step-by-Step Breakdown

### Step 1: Initial Input

```python
idx = torch.tensor([[6109, 3626, 6100, 345]])  # "Every effort moves you"
# Shape: (1, 4) → 1 batch, 4 tokens
```

**Visual Representation:**

```
Batch 1: [6109] [3626] [6100] [345]
           ↓      ↓      ↓      ↓
        "Every" "effort" "moves" "you"
```

---

### Step 2: Context Window Cropping

```python
idx_cond = idx[:, -context_size:]
```

**Why do we need this?**

GPT models have a **maximum context length** they can process (e.g., 1024 tokens for GPT-2). As generation continues, the sequence grows longer. We must crop it to fit within the model's capacity.

**Example Scenario:**

```python
context_size = 5  # Model can only handle 5 tokens

# Generation progresses:
Iteration 1: idx has 4 tokens  → use all 4 (no cropping needed)
Iteration 2: idx has 5 tokens  → use all 5 (at limit)
Iteration 3: idx has 6 tokens  → use last 5 only (cropping happens)
Iteration 4: idx has 7 tokens  → use last 5 only
```

**Visual Example:**

```
Original sequence (10 tokens):
[T1] [T2] [T3] [T4] [T5] [T6] [T7] [T8] [T9] [T10]

Context size = 5, so crop to last 5:
                         [T6] [T7] [T8] [T9] [T10]
                          ↑                      ↑
                      Keep only these for prediction
```

**Important Behavior:**

If the sequence is **shorter** than `context_size`, Python's negative slicing returns **all tokens**:

```python
idx = torch.tensor([[1, 2, 3, 4]])  # 4 tokens
idx_cond = idx[:, -1024:]  # Request last 1024 tokens
# Result: torch.tensor([[1, 2, 3, 4]]) → all 4 tokens returned
```

This is why the function works seamlessly throughout generation—no special case handling needed!

---

### Step 3: Forward Pass Through Model

```python
with torch.no_grad():
    logits = model(idx_cond)  # Shape: (batch, n_tokens, vocab_size)
```

**What are logits?**

Logits are **raw, unnormalized scores** for each token in the vocabulary. Higher scores mean the model thinks that token is more likely to come next.

**Example:**

```python
vocab_size = 50257  # GPT-2 vocabulary

# Input: "Hello, I am"
logits = model(idx_cond)
# Shape: (1, 3, 50257)
#         ↑  ↑    ↑
#      batch tokens vocab

# For each token position, we get 50,257 scores:
logits[0, 0, :] → scores for token after "Hello"
logits[0, 1, :] → scores for token after "I"
logits[0, 2, :] → scores for token after "am"
```

**Visual Representation:**

```
Input Tokens:    ["Hello"]  ["I"]    ["am"]
                     ↓        ↓        ↓
                   [----]   [----]   [----]
                   [----]   [----]   [----]
Model Layers:      [----]   [----]   [----]  (12 transformer blocks)
                   [----]   [----]   [----]
                     ↓        ↓        ↓
Logits:          [50257]  [50257]  [50257]  scores for each vocab token
                   
Each column represents scores for what token should come NEXT
```

---

### Step 4: Focus on Last Token's Logits

```python
logits = logits[:, -1, :]  # Shape: (batch, vocab_size)
```

**Why only the last token?**

We only need to predict the **next single token**, which comes after the last token in our sequence.

**Before:**
```
logits shape: (1, 3, 50257)
              ↑  ↑    ↑
           batch positions vocab
```

**After:**
```
logits shape: (1, 50257)
              ↑    ↑
           batch vocab
           
We extract: logits[:, -1, :] → predictions after the LAST token only
```

**Visual Example:**

```
Full logits tensor:
Position 0: [0.2, -1.5, 0.8, ..., 0.3]  ← After "Hello"
Position 1: [1.1, -0.2, 0.5, ..., -0.8] ← After "I"
Position 2: [2.3, 0.9, -1.1, ..., 1.7]  ← After "am" (LAST position)
             ↑                       ↑
             We ONLY use this row

logits[:, -1, :] = [2.3, 0.9, -1.1, ..., 1.7]
```

---

### Step 5: Convert Logits to Probabilities

```python
probs = torch.softmax(logits, dim=-1)  # Shape: (batch, vocab_size)
```

**What does softmax do?**

Softmax converts raw scores (logits) into a **probability distribution** that sums to 1.0.

**Mathematical Formula:**

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}
$$

Where:
- $z_i$ = logit for token $i$
- $V$ = vocabulary size (50,257 for GPT-2)

**Example:**

```python
# Raw logits for 5 tokens (simplified vocabulary)
logits = torch.tensor([[2.3, 0.9, -1.1, 0.2, 1.7]])

# Apply softmax
probs = torch.softmax(logits, dim=-1)
# Result: tensor([[0.4573, 0.1124, 0.0150, 0.0556, 0.2597]])

# Properties:
sum(probs) = 1.0  ✓
all values between 0 and 1  ✓
highest logit (2.3) → highest probability (0.4573)  ✓
```

**Visual Comparison:**

```
Token ID:    512      1024     3891     7821     9342
             ↓        ↓        ↓        ↓        ↓
Logits:     [2.3]    [0.9]   [-1.1]    [0.2]    [1.7]
             ↓        ↓        ↓        ↓        ↓
Softmax:    [45.7%]  [11.2%]  [1.5%]   [5.6%]  [26.0%]
             ↑                                    ↑
         Most likely                        Second choice
```

---

### Step 6: Greedy Decoding (Select Most Likely Token)

```python
idx_next = torch.argmax(probs, dim=-1, keepdim=True)  # Shape: (batch, 1)
```

**What is Greedy Decoding?**

**Greedy decoding** always picks the token with the **highest probability**. It's called "greedy" because it makes the locally optimal choice at each step without considering future consequences.

**Example:**

```python
probs = tensor([[0.0012, 0.0089, 0.7234, 0.0156, 0.2509, ...]])
                  ↑       ↑       ↑       ↑       ↑
               Token 0  Token 1  Token 2 Token 3  Token 4
               
torch.argmax(probs, dim=-1) = 2  # Token 2 has highest probability (72.34%)

idx_next = torch.tensor([[2]])  # Shape: (1, 1)
```

**Visual Selection Process:**

```
Probability Distribution:
         ┌─────┐
    80% │  █  │
    60% │  █  │
    40% │  █  │  █
    20% │  █  │  █  █  █
     0% └──█──┴──█──█──█──────────►
        "a" "the" "." "," (other tokens)
         ↑
    Select this! (highest bar)
```

**Note on Redundancy:**

```python
# These two approaches give IDENTICAL results:
idx_next = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)  # With softmax
idx_next = torch.argmax(logits, dim=-1)                         # Without softmax

# Why? Softmax is MONOTONIC (preserves order)
# Logits:  [2.3, 0.9, -1.1]  →  argmax = 0
# Probs:   [0.73, 0.18, 0.09] →  argmax = 0 (same!)
```

We include softmax for **conceptual clarity** and because other sampling methods (temperature, top-k, top-p) require actual probabilities.

---

### Step 7: Append New Token to Context

```python
idx = torch.cat((idx, idx_next), dim=1)  # Shape: (batch, n_tokens+1)
```

**Growing the Sequence:**

```python
# Before concatenation:
idx      = torch.tensor([[6109, 3626, 6100, 345]])      # (1, 4)
idx_next = torch.tensor([[521]])                        # (1, 1)

# After concatenation:
idx = torch.tensor([[6109, 3626, 6100, 345, 521]])     # (1, 5)
```

**Visual Growth Over Multiple Iterations:**

```
Iteration 0 (start):
idx = [6109, 3626, 6100, 345]
      "Every effort moves you"

Iteration 1:
idx = [6109, 3626, 6100, 345, 521]
      "Every effort moves you closer"

Iteration 2:
idx = [6109, 3626, 6100, 345, 521, 284]
      "Every effort moves you closer to"

Iteration 3:
idx = [6109, 3626, 6100, 345, 521, 284, 534]
      "Every effort moves you closer to your"

... continues until max_new_tokens reached
```

---

## Complete Generation Example

### Scenario Setup

```python
model = GPTModel(GPT_CONFIG_124M)  # 124M parameter GPT-2
tokenizer = tiktoken.get_encoding("gpt2")

start_text = "The future of AI"
encoded = tokenizer.encode(start_text)  # [464, 2003, 286, 9552]
idx = torch.tensor([encoded])  # Shape: (1, 4)

# Generate 6 new tokens
output = generate_text(
    model=model,
    idx=idx,
    max_new_tokens=6,
    context_size=1024
)
```

### Detailed Iteration Walkthrough

**Iteration 1:**
```
Input:  [464, 2003, 286, 9552]  →  "The future of AI"
         ↓
Crop:   [464, 2003, 286, 9552]  (4 < 1024, use all)
         ↓
Model:  Forward pass
         ↓
Logits: [..., 2.8, ..., 1.3, ...]  (50,257 scores)
         ↓
Softmax: [..., 0.23, ..., 0.08, ...]
         ↓
Argmax: Token ID = 318  ("is")
         ↓
Output: [464, 2003, 286, 9552, 318]  →  "The future of AI is"
```

**Iteration 2:**
```
Input:  [464, 2003, 286, 9552, 318]
         ↓
Crop:   [464, 2003, 286, 9552, 318]  (5 < 1024, use all)
         ↓
Model:  Forward pass
         ↓
Logits: [..., 3.1, ..., 0.9, ...]
         ↓
Softmax: [..., 0.31, ..., 0.05, ...]
         ↓
Argmax: Token ID = 1165  ("bright")
         ↓
Output: [464, 2003, 286, 9552, 318, 1165]  →  "The future of AI is bright"
```

**Iterations 3-6:** (Abbreviated)
```
Iteration 3 → adds token 290 ("and")
Iteration 4 → adds token 1336 ("full")
Iteration 5 → adds token 286 ("of")
Iteration 6 → adds token 5885 ("possibilities")
```

**Final Output:**
```python
output = torch.tensor([[464, 2003, 286, 9552, 318, 1165, 290, 1336, 286, 5885]])
decoded = tokenizer.decode(output[0].tolist())
print(decoded)
# "The future of AI is bright and full of possibilities"
```

---

## Context Window Management in Practice

### Problem: Sequence Exceeds Maximum Context

```python
context_size = 5  # Toy example (GPT-2 uses 1024)
max_new_tokens = 8
start_tokens = [101, 102, 103]  # 3 tokens
```

**Generation Timeline:**

| Iteration | Sequence Length | Tokens in Sequence | Context Used |
|-----------|----------------|-------------------|--------------|
| 0 (start) | 3 | `[101, 102, 103]` | All 3 |
| 1 | 4 | `[101, 102, 103, 104]` | All 4 |
| 2 | 5 | `[101, 102, 103, 104, 105]` | All 5 |
| 3 | 6 | `[101, 102, 103, 104, 105, 106]` | Last 5: `[102, 103, 104, 105, 106]` |
| 4 | 7 | `[101, 102, 103, 104, 105, 106, 107]` | Last 5: `[103, 104, 105, 106, 107]` |
| 5 | 8 | `[101, 102, 103, 104, 105, 106, 107, 108]` | Last 5: `[104, 105, 106, 107, 108]` |

**Visual Sliding Window:**

```
Iteration 0-2 (fits in window):
[101] [102] [103] [104] [105]
 └─────────────────────────┘
   Full sequence used

Iteration 3 (exceeds window):
[101] [102] [103] [104] [105] [106]
       └─────────────────────────┘
       Only last 5 used (101 dropped)

Iteration 4:
[101] [102] [103] [104] [105] [106] [107]
             └─────────────────────────┘
             Only last 5 used (101-102 dropped)
```

This **sliding window** mechanism allows infinite generation while respecting model constraints!

---

## Greedy Decoding: Advantages and Limitations

### ✅ Advantages

1. **Fast**: Single argmax operation, no sampling overhead
2. **Deterministic**: Same input always produces same output
3. **Reproducible**: Perfect for testing and debugging
4. **Simple**: Easy to understand and implement

### ❌ Limitations

1. **Repetitive**: Often generates repetitive text
2. **Boring**: Lacks creativity and diversity
3. **Locally Optimal**: Misses better sequences that start with lower-probability tokens
4. **No Exploration**: Never tries alternative paths

### Example of Repetition Problem

```
Prompt: "The best thing about summer is"

Greedy Output:
"The best thing about summer is the summer. The summer is the best 
time of the year. The summer is the best time of the year."
                                        ↑
                            Gets stuck in a loop!
```

---

## Alternative Sampling Strategies

### 1. Temperature Sampling

**Adjusts probability distribution sharpness:**

```python
def generate_with_temperature(model, idx, max_new_tokens, context_size, temperature=1.0):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature  # Scale by temperature
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)  # Sample from distribution
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

**Temperature Effects:**

```
Temperature = 0.5 (low, more confident):
Probs: [0.85, 0.10, 0.03, 0.02]  →  Very peaked, deterministic

Temperature = 1.0 (default):
Probs: [0.60, 0.25, 0.10, 0.05]  →  Balanced

Temperature = 2.0 (high, more random):
Probs: [0.35, 0.30, 0.20, 0.15]  →  Flattened, more diverse
```

### 2. Top-k Sampling

**Only sample from k most likely tokens:**

```python
# Example with k=3
probs = torch.tensor([0.40, 0.25, 0.15, 0.10, 0.05, 0.03, 0.02])

# Keep only top 3
top_k_probs = torch.tensor([0.40, 0.25, 0.15, 0.00, 0.00, 0.00, 0.00])

# Renormalize
renormalized = top_k_probs / top_k_probs.sum()
# Result: [0.50, 0.3125, 0.1875, 0.0, 0.0, 0.0, 0.0]
```

### 3. Top-p (Nucleus) Sampling

**Sample from smallest set of tokens whose cumulative probability exceeds p:**

```python
# Example with p=0.8
probs = torch.tensor([0.40, 0.25, 0.15, 0.10, 0.05, 0.03, 0.02])
cumsum = torch.tensor([0.40, 0.65, 0.80, 0.90, 0.95, 0.98, 1.00])
                        ↑     ↑     ↑ 
                       Keep these (cumsum ≤ 0.8)

# Tokens 4-7 are zeroed out, then renormalize
```

---

## Real-World Usage Patterns

### Pattern 1: Interactive Chat

```python
conversation_history = []

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    
    # Append user input to history
    conversation_history.append(user_input)
    context = " ".join(conversation_history)
    
    # Encode and generate
    encoded = tokenizer.encode(context)
    idx = torch.tensor([encoded])
    output = generate_text(model, idx, max_new_tokens=50, context_size=1024)
    
    # Decode and display
    response = tokenizer.decode(output[0, len(encoded):].tolist())
    print(f"Bot: {response}")
    conversation_history.append(response)
```

### Pattern 2: Batch Generation

```python
prompts = [
    "The meaning of life is",
    "In the year 2050,",
    "The secret to happiness"
]

# Encode all prompts
encoded_prompts = [tokenizer.encode(p) for p in prompts]

# Pad to same length
max_len = max(len(e) for e in encoded_prompts)
padded = [e + [tokenizer.eot_token] * (max_len - len(e)) for e in encoded_prompts]

# Batch generate
idx = torch.tensor(padded)  # Shape: (3, max_len)
output = generate_text(model, idx, max_new_tokens=30, context_size=1024)

# Decode each
for i, prompt in enumerate(prompts):
    generated = tokenizer.decode(output[i].tolist())
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated}\n")
```

---

## Performance Considerations

### Memory Usage

```python
# For a single generation step:
Sequence length: n
Vocabulary size: V = 50,257
Embedding dimension: d = 768

Memory per forward pass ≈ n × d × 4 bytes (float32)

Example:
n = 512 tokens
Memory ≈ 512 × 768 × 4 = 1.57 MB (just for activations)
```

### Speed Optimization

1. **Use `.eval()` mode**: Disables dropout, enables inference optimizations
2. **Use `torch.no_grad()`**: Prevents gradient computation (saves memory)
3. **Batch processing**: Generate multiple sequences in parallel
4. **KV-cache**: Cache key-value pairs from previous tokens (advanced)

---

## Common Pitfalls and Solutions

### Pitfall 1: Forgetting `.eval()` Mode

```python
# ❌ WRONG
output = generate_text(model, idx, max_new_tokens=50, context_size=1024)

# ✅ CORRECT
model.eval()  # Disable dropout and batch normalization
output = generate_text(model, idx, max_new_tokens=50, context_size=1024)
```

### Pitfall 2: Generating Too Long Without Stopping

```python
# ❌ Can generate forever
output = generate_text(model, idx, max_new_tokens=10000, context_size=1024)

# ✅ Add stopping criteria
def generate_with_stop(model, idx, max_new_tokens, context_size, stop_tokens):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)
        
        # Check for stop tokens
        if idx_next.item() in stop_tokens:
            break
            
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

### Pitfall 3: Not Handling Device (CPU vs GPU)

```python
# ✅ Ensure tensors are on same device as model
device = next(model.parameters()).device
idx = idx.to(device)
output = generate_text(model, idx, max_new_tokens=50, context_size=1024)
```

---

## Summary

### Key Takeaways

1. **Autoregressive generation** produces one token at a time using previous tokens as context
2. **Context cropping** (`idx[:, -context_size:]`) enables generation beyond model's maximum length
3. **Greedy decoding** (argmax) is simple but can be repetitive
4. **Softmax converts logits → probabilities** (though argmax works on both)
5. **The loop continues** for `max_new_tokens` iterations, growing the sequence each time

### The Core Loop Visualized

```
    ┌─────────────────────────────────────┐
    │  Start with prompt tokens           │
    └─────────────────┬───────────────────┘
                      │
    ┌─────────────────▼───────────────────┐
    │  Crop to last context_size tokens   │◄──────┐
    └─────────────────┬───────────────────┘       │
                      │                            │
    ┌─────────────────▼───────────────────┐       │
    │  Forward pass through model         │       │
    └─────────────────┬───────────────────┘       │
                      │                            │
    ┌─────────────────▼───────────────────┐       │
    │  Extract logits for last position   │       │
    └─────────────────┬───────────────────┘       │
                      │                            │
    ┌─────────────────▼───────────────────┐       │
    │  Softmax → probabilities            │       │
    └─────────────────┬───────────────────┘       │
                      │                            │
    ┌─────────────────▼───────────────────┐       │
    │  Argmax → select most likely token  │       │
    └─────────────────┬───────────────────┘       │
                      │                            │
    ┌─────────────────▼───────────────────┐       │
    │  Append token to sequence           │       │
    └─────────────────┬───────────────────┘       │
                      │                            │
    ┌─────────────────▼───────────────────┐       │
    │  Reached max_new_tokens?            │       │
    │         No ──────────────────────────┘       │
    │         Yes                                  │
    └─────────────────┬───────────────────────────┘
                      │
    ┌─────────────────▼───────────────────┐
    │  Return complete sequence           │
    └─────────────────────────────────────┘
```

### What Makes This Function Powerful

- **Simplicity**: Just 15 lines of code generate unlimited text
- **Flexibility**: Works with any transformer model architecture
- **Scalability**: Handles sequences longer than model's context window
- **Efficiency**: Context cropping prevents memory explosion

This is the foundation of how ChatGPT, GPT-4, and all autoregressive language models generate text!
