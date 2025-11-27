# Advanced Text Generation with Temperature and Top-K Sampling

## Overview

The `generate()` function is an **enhanced text generation method** that improves upon simple greedy decoding by adding **temperature scaling** and **top-k sampling** to control randomness and diversity in LLM-generated text.

Unlike `generate_text_simple()` which always picks the most likely token (greedy decoding), this function offers **three generation strategies**:

1. **Greedy Decoding** (temperature = 0.0)
2. **Temperature Sampling** (temperature > 0.0, top_k = None)
3. **Top-K + Temperature Sampling** (temperature > 0.0, top_k = value)

---

## Function Signature

```python
def generate(model, idx, max_new_tokens, context_size, 
             temperature=0.0, top_k=None, eos_id=None):
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | Required | The trained GPT model |
| `idx` | `torch.Tensor` | Required | Input token IDs, shape `(batch_size, seq_len)` |
| `max_new_tokens` | `int` | Required | Maximum number of tokens to generate |
| `context_size` | `int` | Required | Model's maximum context window (e.g., 256) |
| `temperature` | `float` | `0.0` | Controls randomness (0.0 = greedy, >1 = more random) |
| `top_k` | `int` or `None` | `None` | If set, only sample from top-k most likely tokens |
| `eos_id` | `int` or `None` | `None` | End-of-sequence token ID to stop generation early |

---

## Step-by-Step Breakdown

### Step 1: Context Window Cropping

```python
idx_cond = idx[:, -context_size:]
```

**What it does:**
- Crops the input sequence to the last `context_size` tokens
- Prevents exceeding the model's maximum context length

**Example:**
```python
# If idx has 300 tokens but context_size = 256
idx = tensor([[1, 2, 3, ..., 300]])  # Shape: (1, 300)
idx_cond = idx[:, -256:]              # Shape: (1, 256) - last 256 tokens
```

**Why it matters:**
- Positional embeddings are only trained up to `context_size`
- Longer sequences would cause index errors

---

### Step 2: Forward Pass (No Gradient Tracking)

```python
with torch.no_grad():
    logits = model(idx_cond)
logits = logits[:, -1, :]
```

**What it does:**
1. **`torch.no_grad()`**: Disables gradient computation for efficiency
2. **Forward pass**: Gets raw logits for all tokens, shape `(batch, seq_len, vocab_size)`
3. **Extract last token**: `logits[:, -1, :]` → shape `(batch, vocab_size)`

**Visual representation:**
```
Input:  [Every, effort, moves, you]
         ↓       ↓       ↓      ↓
Model: [logits, logits, logits, logits]
                                  ↓
                         We only use this one
                         (predicts next token)
```

**Example:**
```python
# Before: logits.shape = (1, 4, 50257)  # 4 tokens, 50257 vocab
# After:  logits.shape = (1, 50257)      # Only last token's predictions
```

---

### Step 3: Top-K Filtering (Optional)

```python
if top_k is not None:
    top_logits, _ = torch.topk(logits, top_k)
    min_val = top_logits[:, -1]
    logits = torch.where(logits < min_val, 
                         torch.tensor(float("-inf")).to(logits.device), 
                         logits)
```

**What it does:**
- Restricts token selection to the **top-k most likely tokens**
- Sets all other tokens' logits to `-inf` (0% probability after softmax)

**Step-by-step example with `top_k=3`:**

```python
# Original logits (simplified vocab)
logits = tensor([4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79])
#                  ↑                    ↑                        ↑
#               closer               forward                  toward

# Step 1: Get top-3 logits
top_logits = tensor([6.75, 6.28, 4.51])  # forward, toward, closer
min_val = 4.51  # The 3rd highest value

# Step 2: Mask everything below 4.51
new_logits = tensor([4.51, -inf, -inf, 6.75, -inf, -inf, -inf, 6.28, -inf])
#                    keep                keep                  keep

# Step 3: After softmax, only these 3 tokens have non-zero probability
probas = tensor([0.169, 0.0, 0.0, 0.516, 0.0, 0.0, 0.0, 0.315, 0.0])
#                closer           forward                toward
```

**Visual representation:**

```
Before Top-K (all tokens considered):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
█████ closer
██ every
█ effort
████████ forward   ← Top 1
███ inches
█ moves
█ pizza
███████ toward     ← Top 3
██ you
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

After Top-K=3 (only top-3 kept):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
█████ closer       ← Kept
                   ← Masked (-inf)
                   ← Masked (-inf)
████████ forward   ← Kept
                   ← Masked (-inf)
                   ← Masked (-inf)
                   ← Masked (-inf)
███████ toward     ← Kept
                   ← Masked (-inf)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Why use Top-K?**
- ✅ Prevents selecting nonsensical low-probability tokens like "pizza"
- ✅ Maintains diversity among reasonable tokens
- ✅ Reduces computational waste on unlikely candidates

---

### Step 4: Temperature Scaling + Sampling

```python
if temperature > 0.0:
    logits = logits / temperature
    probas = torch.softmax(logits, dim=-1)
    idx_next = torch.multinomial(probas, num_samples=1)
```

**What it does:**
- **Temperature scaling**: Divides logits by temperature before softmax
- **Multinomial sampling**: Randomly selects token based on probabilities

**Temperature effects:**

| Temperature | Effect | Distribution | Use Case |
|-------------|--------|--------------|----------|
| **0.1 - 0.5** | Sharper | Strongly favors top tokens | Factual, consistent text |
| **1.0** | Neutral | Original probabilities | Balanced generation |
| **1.5 - 2.0** | Softer | More uniform distribution | Creative, diverse text |
| **5.0+** | Very soft | Nearly uniform | Highly random, experimental |

**Mathematical example:**

```python
# Original logits
logits = tensor([4.51, 6.75, 6.28])  # closer, forward, toward

# Temperature = 0.5 (sharper - more confident)
scaled = logits / 0.5 = tensor([9.02, 13.5, 12.56])
probas = softmax(scaled) = tensor([0.018, 0.735, 0.247])
# "forward" dominates even more (73.5%)

# Temperature = 1.0 (original)
scaled = logits / 1.0 = tensor([4.51, 6.75, 6.28])
probas = softmax(scaled) = tensor([0.090, 0.516, 0.394])
# "forward" is most likely (51.6%)

# Temperature = 5.0 (softer - more random)
scaled = logits / 5.0 = tensor([0.90, 1.35, 1.26])
probas = softmax(scaled) = tensor([0.256, 0.403, 0.341])
# More uniform distribution (40.3% vs 34.1% vs 25.6%)
```

**Visual comparison:**

```
Temperature = 0.5 (Sharp - Confident)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
█ closer (1.8%)
████████████████████████████████ forward (73.5%)
███████████ toward (24.7%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Temperature = 1.0 (Balanced)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
████ closer (9.0%)
█████████████████████ forward (51.6%)
███████████████ toward (39.4%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Temperature = 5.0 (Soft - Creative)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
██████████ closer (25.6%)
███████████████ forward (40.3%)
████████████ toward (34.1%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**`torch.multinomial()` behavior:**
```python
# With probas = [0.09, 0.516, 0.394]
# Run 1000 times:
# - "closer" selected ~90 times (9%)
# - "forward" selected ~516 times (51.6%)
# - "toward" selected ~394 times (39.4%)

# This adds VARIETY - different runs produce different tokens
```

---

### Step 5: Greedy Decoding (Fallback)

```python
else:
    idx_next = torch.argmax(logits, dim=-1, keepdim=True)
```

**What it does:**
- When `temperature = 0.0`, falls back to **greedy decoding**
- Always selects the token with the highest logit value
- **Deterministic**: Same input always produces same output

**Example:**
```python
logits = tensor([4.51, 6.75, 6.28])  # closer, forward, toward
idx_next = torch.argmax(logits)      # Returns: 1 (forward)
# Always picks "forward" - no randomness
```

**When to use:**
- Factual question answering
- Code generation (need deterministic output)
- Reproducible experiments

---

### Step 6: Early Stopping (EOS Token)

```python
if eos_id is not None and idx_next.item() == eos_id:
    break
```

**What it does:**
- Stops generation if the model produces the **end-of-sequence token**
- Prevents generating unnecessary padding tokens

**Example:**
```python
# GPT-2's <|endoftext|> token has ID 50256
tokenizer = tiktoken.get_encoding("gpt2")
eos_id = tokenizer.encode("<|endoftext|>")[0]  # 50256

generate(
    model=model,
    idx=encoded_input,
    max_new_tokens=100,
    context_size=256,
    eos_id=eos_id  # Will stop early if model generates <|endoftext|>
)

# If model generates:
# "The story ends here.<|endoftext|>"
#                       ↑ Generation stops here, even if max_new_tokens not reached
```

**Why it matters:**
- More efficient (doesn't waste computation)
- Natural stopping point for coherent text
- Mimics real conversational turn-taking

---

### Step 7: Token Concatenation

```python
idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)
```

**What it does:**
- Appends the newly generated token to the sequence
- Grows the context for the next iteration

**Example progression:**
```python
# Iteration 1:
idx = tensor([[6109, 3626, 6100, 345]])  # "Every effort moves you"
idx_next = tensor([[1640]])               # "forward"
idx = tensor([[6109, 3626, 6100, 345, 1640]])  # "Every effort moves you forward"

# Iteration 2:
idx_next = tensor([[11]])                 # ","
idx = tensor([[6109, 3626, 6100, 345, 1640, 11]])  # "Every effort moves you forward,"

# ...continues for max_new_tokens iterations
```

---

## Complete Generation Strategies Comparison

### Strategy 1: Greedy Decoding (Deterministic)

```python
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=15,
    context_size=256,
    temperature=0.0,  # Greedy decoding
    top_k=None
)
# Output: "Every effort moves you forward to the next level of the game."
# Always the same output for same input
```

**Characteristics:**
- ✅ Deterministic and reproducible
- ✅ Grammatically correct
- ❌ Repetitive and boring
- ❌ No creativity

---

### Strategy 2: Temperature Sampling Only

```python
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=15,
    context_size=256,
    temperature=1.5,  # Creative sampling
    top_k=None        # Consider all tokens
)
# Output: "Every effort moves you inches toward pizza closer happiness..."
# More diverse but sometimes nonsensical
```

**Characteristics:**
- ✅ Creative and diverse
- ✅ Non-repetitive
- ❌ Can generate nonsense ("pizza")
- ❌ Less coherent

---

### Strategy 3: Top-K + Temperature (Recommended)

```python
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=15,
    context_size=256,
    temperature=1.0,  # Balanced randomness
    top_k=25          # Only consider top-25 tokens
)
# Output: "Every effort moves you closer to your goals and dreams."
# Diverse yet coherent
```

**Characteristics:**
- ✅ Good balance of diversity and coherence
- ✅ Prevents nonsensical tokens
- ✅ Still creative within reasonable bounds
- ✅ **Industry standard** for production LLMs

---

## Recommended Parameter Settings

### For Different Use Cases

| Use Case | Temperature | Top-K | Description |
|----------|-------------|-------|-------------|
| **Code Generation** | 0.0 | None | Deterministic, precise |
| **Factual QA** | 0.3 | 10-20 | Slightly creative but accurate |
| **Chatbot** | 0.7-1.0 | 25-50 | Natural conversation |
| **Creative Writing** | 1.2-1.5 | 50-100 | More imaginative |
| **Brainstorming** | 1.5-2.0 | None | Maximum diversity |
| **Poetry/Art** | 1.8-2.5 | 100+ | Experimental |

---

## Practical Examples

### Example 1: Conservative Generation

```python
torch.manual_seed(123)

output = generate(
    model=model,
    idx=text_to_token_ids("The capital of France is", tokenizer),
    max_new_tokens=10,
    context_size=256,
    temperature=0.3,  # Low temperature = high confidence
    top_k=10          # Only top-10 tokens
)

print(token_ids_to_text(output, tokenizer))
# Output: "The capital of France is Paris, which is known for"
# Accurate and factual
```

---

### Example 2: Creative Generation

```python
torch.manual_seed(456)

output = generate(
    model=model,
    idx=text_to_token_ids("Once upon a time", tokenizer),
    max_new_tokens=30,
    context_size=256,
    temperature=1.5,  # High temperature = more creative
    top_k=50          # Wider selection
)

print(token_ids_to_text(output, tokenizer))
# Output: "Once upon a time, in a kingdom hidden beneath crystalline waves,
#          there lived a young dreamer who collected forgotten melodies..."
# More imaginative and varied
```

---

### Example 3: With Early Stopping

```python
eos_token_id = tokenizer.encode("<|endoftext|>", allowed_special={'<|endoftext|>'})[0]

output = generate(
    model=model,
    idx=text_to_token_ids("To summarize:", tokenizer),
    max_new_tokens=100,
    context_size=256,
    temperature=0.7,
    top_k=30,
    eos_id=eos_token_id  # Stop when model generates <|endoftext|>
)

print(token_ids_to_text(output, tokenizer))
# Output: "To summarize: The study shows that regular exercise improves
#          mental health and cognitive function.<|endoftext|>"
# Stops naturally when model signals completion
```

---

## Behind the Scenes: Full Iteration Example

Let's trace through one complete iteration with **top_k=3** and **temperature=1.0**:

```python
# Given context: "Every effort moves you"
# Input: idx = tensor([[6109, 3626, 6100, 345]])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Step 1: Context cropping (no change, under 256 tokens)
idx_cond = tensor([[6109, 3626, 6100, 345]])  # "Every effort moves you"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Step 2: Forward pass
with torch.no_grad():
    logits = model(idx_cond)  # Shape: (1, 4, 50257)

logits = logits[:, -1, :]     # Shape: (1, 50257) - predictions for next token

# Raw logits for sample vocab:
# closer=4.51, forward=6.75, toward=6.28, inches=1.63, pizza=-1.89, ...

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Step 3: Top-K filtering (k=3)
top_logits, _ = torch.topk(logits, 3)
# top_logits = tensor([[6.75, 6.28, 4.51]])  # forward, toward, closer
min_val = 4.51

logits = torch.where(logits < 4.51, torch.tensor(float("-inf")), logits)
# Now: closer=4.51, forward=6.75, toward=6.28, everything_else=-inf

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Step 4: Temperature scaling (T=1.0)
logits = logits / 1.0  # No change since temperature=1.0

probas = torch.softmax(logits, dim=-1)
# probas = tensor([[..., 0.090, ..., 0.516, ..., 0.394, ...]])
#                     closer      forward      toward

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Step 5: Multinomial sampling
idx_next = torch.multinomial(probas, num_samples=1)
# Randomly samples: 51.6% chance "forward", 39.4% "toward", 9.0% "closer"
# Let's say it samples: tensor([[forward_id]])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Step 6: EOS check (eos_id=None, skip)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Step 7: Concatenation
idx = torch.cat((idx, idx_next), dim=1)
# idx = tensor([[6109, 3626, 6100, 345, forward_id]])
# "Every effort moves you forward"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Loop continues for max_new_tokens iterations...
```

---

## Key Differences from `generate_text_simple()`

| Feature | `generate_text_simple()` | `generate()` |
|---------|-------------------------|--------------|
| **Decoding** | Greedy only (argmax) | Greedy, Temperature, Top-K |
| **Randomness** | None (deterministic) | Controllable via temperature |
| **Token Selection** | All tokens considered | Can restrict to top-k |
| **Early Stopping** | No | Yes (via eos_id) |
| **Use Case** | Basic demo | Production-ready |
| **Diversity** | Low (repetitive) | High (adjustable) |
| **Coherence** | High | Adjustable (temperature + top_k) |

---

## Common Pitfalls and Solutions

### Pitfall 1: Temperature = 0 with Top-K

```python
# ❌ Don't do this:
generate(model, idx, 15, 256, temperature=0.0, top_k=25)
# Top-K filtering is ignored because temperature=0 uses argmax

# ✅ Do this instead:
generate(model, idx, 15, 256, temperature=0.7, top_k=25)
```

---

### Pitfall 2: Temperature Too High

```python
# ❌ Too random:
generate(model, idx, 15, 256, temperature=5.0, top_k=None)
# Output: "Every effort moves you pizza quantum umbrella elephant..."

# ✅ Reasonable temperature:
generate(model, idx, 15, 256, temperature=1.2, top_k=50)
# Output: "Every effort moves you closer to achieving your dreams and goals..."
```

---

### Pitfall 3: Top-K Too Small

```python
# ❌ Too restrictive:
generate(model, idx, 15, 256, temperature=1.0, top_k=2)
# Very limited vocabulary, repetitive

# ✅ Balanced:
generate(model, idx, 15, 256, temperature=1.0, top_k=25)
# Good balance of diversity and coherence
```

---

### Pitfall 4: Wrong Device for `-inf` Tensor

```python
# ❌ Device mismatch:
logits = torch.where(logits < min_val, torch.tensor(float("-inf")), logits)
# If logits is on GPU, this creates CPU tensor → error

# ✅ Match device:
logits = torch.where(logits < min_val, 
                     torch.tensor(float("-inf")).to(logits.device), 
                     logits)
```

---

## Performance Considerations

### Memory Usage

```python
# Memory primarily determined by:
# 1. Batch size (in idx)
# 2. Vocabulary size (50,257 for GPT-2)
# 3. Context size (256 in your config)

# Top-K reduces computation but not memory (still compute full logits first)
```

### Speed Optimization

```python
# Faster generation:
# 1. Use lower context_size if possible
# 2. Smaller top_k values (less sorting)
# 3. Batch multiple sequences together
# 4. Use GPU (CUDA) instead of CPU

# Example: Batch generation
idx_batch = torch.cat([
    text_to_token_ids("First prompt", tokenizer),
    text_to_token_ids("Second prompt", tokenizer),
], dim=0)  # Shape: (2, seq_len)

# Generates for both prompts simultaneously
output = generate(model, idx_batch, 15, 256, temperature=0.7, top_k=25)
```

---

## Summary

The `generate()` function provides **three key improvements** over simple greedy decoding:

1. **Top-K Sampling**: Restricts selection to k most likely tokens
   - Prevents nonsensical outputs
   - Maintains coherence while adding diversity

2. **Temperature Scaling**: Controls randomness
   - Low (0.1-0.5): Confident, deterministic-like
   - Medium (0.7-1.2): Balanced, natural
   - High (1.5+): Creative, experimental

3. **Early Stopping**: Respects end-of-sequence tokens
   - More efficient
   - Natural completion

**Recommended Settings for Most Use Cases:**
```python
generate(
    model=model,
    idx=encoded_input,
    max_new_tokens=50,
    context_size=256,
    temperature=0.7,    # Slightly creative
    top_k=25,           # Reasonable token pool
    eos_id=eos_token_id # Stop naturally
)
```

This produces **coherent yet diverse** text suitable for chatbots, content generation, and interactive applications.
