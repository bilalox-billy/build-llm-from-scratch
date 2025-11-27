# Tokenizer Conversion Functions: Complete Guide

## Overview

These two utility functions provide a convenient interface for converting between human-readable text and the numerical token representations that GPT models understand. They handle the critical transformations needed for model input/output.

---

## Table of Contents

1. [What is tiktoken?](#what-is-tiktoken)
2. [Function 1: `text_to_token_ids()`](#function-1-text_to_token_ids)
3. [Function 2: `token_ids_to_text()`](#function-2-token_ids_to_text)
4. [Complete Workflow Examples](#complete-workflow-examples)
5. [Special Tokens Explained](#special-tokens-explained)
6. [Batch Dimension Deep Dive](#batch-dimension-deep-dive)
7. [Common Use Cases](#common-use-cases)
8. [Troubleshooting](#troubleshooting)

---

## What is tiktoken?

**tiktoken** is OpenAI's official tokenizer library for GPT models. It's written in Rust for speed and provides the exact tokenization used by GPT-2, GPT-3, and GPT-4.

### Key Features

- **Fast**: 3-6x faster than other Python tokenizers
- **Accurate**: Matches OpenAI's production tokenization exactly
- **Multiple Encodings**: Supports different GPT model vocabularies

### Available Encodings

```python
import tiktoken

# GPT-2 / GPT-3
tokenizer = tiktoken.get_encoding("gpt2")        # 50,257 tokens
tokenizer = tiktoken.get_encoding("p50k_base")   # 50,281 tokens

# GPT-3.5-turbo / GPT-4
tokenizer = tiktoken.get_encoding("cl100k_base") # 100,277 tokens
```

---

## Function 1: `text_to_token_ids()`

### Purpose

Converts human-readable text into a PyTorch tensor of token IDs that can be fed into a GPT model.

### Function Code

```python
import tiktoken
import torch

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special = {'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor
```

---

### Step-by-Step Breakdown

#### Step 1: Encode Text to Token IDs

```python
encoded = tokenizer.encode(text, allowed_special = {'<|endoftext|>'})
```

**What happens:**
- Takes input text string
- Splits it into subword tokens using Byte Pair Encoding (BPE)
- Converts each token to its vocabulary index (integer)
- Returns a Python list of integers

**Example:**

```python
tokenizer = tiktoken.get_encoding("gpt2")
text = "Hello, world!"
encoded = tokenizer.encode(text)
print(encoded)
# Output: [15496, 11, 995, 0]
```

**Visual Representation:**

```
Input Text: "Hello, world!"
              ↓
         [Tokenization]
              ↓
    Token 0: "Hello"  → ID: 15496
    Token 1: ","      → ID: 11
    Token 2: " world" → ID: 995
    Token 3: "!"      → ID: 0
              ↓
    Output: [15496, 11, 995, 0]
```

#### Step 2: The `allowed_special` Parameter

```python
allowed_special = {'<|endoftext|>'}
```

**What are special tokens?**

Special tokens are control tokens that have special meanings:

| Token | Purpose | Example Use |
|-------|---------|-------------|
| `<|endoftext|>` | Document separator | Marks end of text/document |
| `<|fim_prefix|>` | Fill-in-middle prefix | Code completion |
| `<|fim_middle|>` | Fill-in-middle middle | Code completion |
| `<|fim_suffix|>` | Fill-in-middle suffix | Code completion |

**Why `allowed_special`?**

By default, tiktoken **raises an error** if it encounters special tokens in text. This prevents accidental inclusion of special tokens that could confuse the model.

```python
# ❌ WITHOUT allowed_special (will raise error)
text = "This is text.<|endoftext|>More text."
encoded = tokenizer.encode(text)  # ERROR!

# ✅ WITH allowed_special (works correctly)
encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
# Output: [1212, 318, 2420, 13, 50256, 5167, 2420, 13]
#                              ↑
#                        <|endoftext|> token ID
```

**Visual Example:**

```
Text: "First document.<|endoftext|>Second document."

Without allowed_special:
    ❌ RAISES ERROR

With allowed_special={'<|endoftext|>'}:
    ✅ "First"     → 5962
       " document" → 3188
       "."         → 13
       "<|endoftext|>" → 50256  (special token ID)
       "Second"    → 12211
       " document" → 3188
       "."         → 13
```

#### Step 3: Convert to PyTorch Tensor

```python
encoded_tensor = torch.tensor(encoded)
```

**Why PyTorch tensors?**

PyTorch models require tensor inputs, not Python lists. Tensors enable:
- GPU acceleration
- Automatic differentiation (gradients)
- Batch processing
- Integration with neural network operations

**Example:**

```python
encoded = [15496, 11, 995, 0]  # Python list
encoded_tensor = torch.tensor(encoded)
print(encoded_tensor)
# Output: tensor([15496, 11, 995, 0])
print(type(encoded_tensor))
# Output: <class 'torch.Tensor'>
```

#### Step 4: Add Batch Dimension with `unsqueeze(0)`

```python
encoded_tensor = encoded_tensor.unsqueeze(0)
```

**What is `unsqueeze(0)`?**

It adds a new dimension at position 0, converting a 1D tensor into a 2D tensor with shape `(1, n_tokens)`.

**Why do we need a batch dimension?**

Neural networks process data in **batches** for efficiency. Even when processing a single input, models expect shape `(batch_size, sequence_length)`.

**Visual Transformation:**

```
Before unsqueeze(0):
Shape: (4,)
tensor([15496, 11, 995, 0])
         ↓     ↓   ↓   ↓
       Token0 T1  T2  T3

After unsqueeze(0):
Shape: (1, 4)
tensor([[15496, 11, 995, 0]])
          ↑                ↑
        Batch 1 with 4 tokens
```

**Detailed Example:**

```python
# 1D tensor (no batch dimension)
tensor_1d = torch.tensor([15496, 11, 995, 0])
print(tensor_1d.shape)  # torch.Size([4])
print(tensor_1d.dim())  # 1 dimension

# Add batch dimension
tensor_2d = tensor_1d.unsqueeze(0)
print(tensor_2d.shape)  # torch.Size([1, 4])
print(tensor_2d.dim())  # 2 dimensions

# This matches what GPT models expect
model_input = tensor_2d  # Shape: (batch_size=1, seq_len=4)
```

---

### Complete Function Flow

```python
text = "Hello, world!"

# Step 1: Encode
encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
# Result: [15496, 11, 995, 0] (Python list)

# Step 2: Convert to tensor
encoded_tensor = torch.tensor(encoded)
# Result: tensor([15496, 11, 995, 0]) (1D tensor)

# Step 3: Add batch dimension
encoded_tensor = encoded_tensor.unsqueeze(0)
# Result: tensor([[15496, 11, 995, 0]]) (2D tensor)

# Return
return encoded_tensor
```

**Visual Pipeline:**

```
    "Hello, world!"
          ↓
    [tokenize]
          ↓
    [15496, 11, 995, 0]  ← Python list
          ↓
    [torch.tensor()]
          ↓
    tensor([15496, 11, 995, 0])  ← 1D tensor, shape (4,)
          ↓
    [unsqueeze(0)]
          ↓
    tensor([[15496, 11, 995, 0]])  ← 2D tensor, shape (1, 4)
          ↓
    [ready for model]
```

---

## Function 2: `token_ids_to_text()`

### Purpose

Converts a PyTorch tensor of token IDs back into human-readable text. This is the inverse operation of `text_to_token_ids()`.

### Function Code

```python
def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # Remove batch dimension
    return tokenizer.decode(flat.tolist())
```

---

### Step-by-Step Breakdown

#### Step 1: Remove Batch Dimension with `squeeze(0)`

```python
flat = token_ids.squeeze(0)
```

**What is `squeeze(0)`?**

It removes a dimension of size 1 at position 0, converting a 2D tensor `(1, n_tokens)` back to a 1D tensor `(n_tokens,)`.

**Why remove the batch dimension?**

The tokenizer's `decode()` method expects a simple list of token IDs, not a batched tensor. We need to "flatten" the tensor back to 1D.

**Visual Transformation:**

```
Before squeeze(0):
Shape: (1, 4)
tensor([[15496, 11, 995, 0]])
          ↑                ↑
        1 batch, 4 tokens

After squeeze(0):
Shape: (4,)
tensor([15496, 11, 995, 0])
         ↓     ↓   ↓   ↓
       Simple 1D sequence
```

**Detailed Example:**

```python
# Model output (2D tensor with batch dimension)
token_ids = torch.tensor([[15496, 11, 995, 0]])
print(token_ids.shape)  # torch.Size([1, 4])
print(token_ids)
# tensor([[15496, 11, 995, 0]])

# Remove batch dimension
flat = token_ids.squeeze(0)
print(flat.shape)  # torch.Size([4])
print(flat)
# tensor([15496, 11, 995, 0])
```

**Important Note:**

`squeeze(0)` only removes dimension 0 if it has size 1. If you have multiple batches, it won't remove anything:

```python
# Multiple batches
multi_batch = torch.tensor([[15496, 11], [995, 0]])
print(multi_batch.shape)  # torch.Size([2, 2])

squeezed = multi_batch.squeeze(0)
print(squeezed.shape)  # torch.Size([2, 2]) - unchanged!
```

#### Step 2: Convert Tensor to Python List

```python
flat.tolist()
```

**Why `.tolist()`?**

The tokenizer's `decode()` method expects a Python list of integers, not a PyTorch tensor. We must convert the tensor to a native Python data structure.

**Example:**

```python
flat = torch.tensor([15496, 11, 995, 0])
python_list = flat.tolist()

print(type(flat))        # <class 'torch.Tensor'>
print(type(python_list)) # <class 'list'>
print(python_list)       # [15496, 11, 995, 0]
```

#### Step 3: Decode Token IDs to Text

```python
tokenizer.decode(flat.tolist())
```

**What happens:**
- Takes list of token IDs
- Looks up each ID in the vocabulary
- Converts each ID back to its subword token
- Concatenates all tokens into a string
- Returns the original text

**Example:**

```python
token_ids = [15496, 11, 995, 0]
text = tokenizer.decode(token_ids)
print(text)
# Output: "Hello, world!"
```

**Visual Representation:**

```
Token IDs: [15496, 11, 995, 0]
              ↓    ↓   ↓   ↓
         [Vocabulary Lookup]
              ↓    ↓   ↓   ↓
    ID 15496 → "Hello"
    ID 11    → ","
    ID 995   → " world"
    ID 0     → "!"
              ↓
      [Concatenate]
              ↓
      "Hello, world!"
```

---

### Complete Function Flow

```python
token_ids = torch.tensor([[15496, 11, 995, 0]])

# Step 1: Remove batch dimension
flat = token_ids.squeeze(0)
# Result: tensor([15496, 11, 995, 0])

# Step 2: Convert to Python list
token_list = flat.tolist()
# Result: [15496, 11, 995, 0]

# Step 3: Decode to text
text = tokenizer.decode(token_list)
# Result: "Hello, world!"

# Return
return text
```

**Visual Pipeline:**

```
tensor([[15496, 11, 995, 0]])  ← 2D tensor, shape (1, 4)
          ↓
    [squeeze(0)]
          ↓
tensor([15496, 11, 995, 0])  ← 1D tensor, shape (4,)
          ↓
    [.tolist()]
          ↓
[15496, 11, 995, 0]  ← Python list
          ↓
    [tokenizer.decode()]
          ↓
    "Hello, world!"  ← Human-readable text
```

---

## Complete Workflow Examples

### Example 1: Simple Text Conversion

```python
import tiktoken
import torch

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Original text
text = "Artificial intelligence is amazing!"

# Text → Token IDs
token_ids = text_to_token_ids(text, tokenizer)
print("Token IDs:", token_ids)
print("Shape:", token_ids.shape)

# Token IDs → Text
recovered_text = token_ids_to_text(token_ids, tokenizer)
print("Recovered:", recovered_text)
```

**Output:**

```
Token IDs: tensor([[8001, 9542, 4430, 318, 4998, 0]])
Shape: torch.Size([1, 6])
Recovered: Artificial intelligence is amazing!
```

**Visual Flow:**

```
"Artificial intelligence is amazing!"
                ↓
    [text_to_token_ids()]
                ↓
tensor([[8001, 9542, 4430, 318, 4998, 0]])
                ↓
   [token_ids_to_text()]
                ↓
"Artificial intelligence is amazing!"
```

---

### Example 2: Multi-Sentence Text with Special Tokens

```python
text = "First sentence.<|endoftext|>Second sentence."

# Convert with special token handling
token_ids = text_to_token_ids(text, tokenizer)
print("Token IDs:", token_ids)
print("Shape:", token_ids.shape)

# Decode back
recovered = token_ids_to_text(token_ids, tokenizer)
print("Recovered:", recovered)
```

**Output:**

```
Token IDs: tensor([[5962, 6827, 13, 50256, 12211, 6827, 13]])
Shape: torch.Size([1, 7])
Recovered: First sentence.<|endoftext|>Second sentence.
```

**Token Breakdown:**

| Token | ID | Text |
|-------|-------|------|
| 0 | 5962 | "First" |
| 1 | 6827 | " sentence" |
| 2 | 13 | "." |
| 3 | 50256 | `<|endoftext|>` |
| 4 | 12211 | "Second" |
| 5 | 6827 | " sentence" |
| 6 | 13 | "." |

---

### Example 3: Model Integration

```python
# Prepare input
prompt = "The future of AI"
input_ids = text_to_token_ids(prompt, tokenizer)
print("Input shape:", input_ids.shape)

# Generate with model
model.eval()
with torch.no_grad():
    output_ids = generate_text_simple(
        model=model,
        idx=input_ids,
        max_new_tokens=10,
        context_size=256
    )

print("Output shape:", output_ids.shape)

# Convert output back to text
generated_text = token_ids_to_text(output_ids, tokenizer)
print("Generated:", generated_text)
```

**Output:**

```
Input shape: torch.Size([1, 4])
Output shape: torch.Size([1, 14])
Generated: The future of AI is bright and full of endless possibilities.
```

**Complete Flow:**

```
    "The future of AI"
            ↓
    [text_to_token_ids]
            ↓
    tensor([[464, 2003, 286, 9552]])  ← shape (1, 4)
            ↓
    [GPT Model Generation]
            ↓
    tensor([[464, 2003, 286, 9552, 318, 6016, 290, 1336, 286, 6079]])  ← shape (1, 14)
            ↓
    [token_ids_to_text]
            ↓
    "The future of AI is bright and full of endless possibilities."
```

---

## Special Tokens Explained

### What Are Special Tokens?

Special tokens are reserved tokens with predefined meanings in the tokenizer vocabulary. They don't represent regular words but serve control purposes.

### GPT-2 Special Tokens

| Token | ID | Purpose |
|-------|-----|---------|
| `<|endoftext|>` | 50256 | End of document/separator |

**Note:** GPT-2 has only one special token. GPT-3.5 and GPT-4 have more.

### GPT-3.5/GPT-4 Special Tokens

| Token | ID | Purpose |
|-------|-----|---------|
| `<|endoftext|>` | 100257 | End of document |
| `<|fim_prefix|>` | 100258 | Fill-in-middle: prefix part |
| `<|fim_middle|>` | 100259 | Fill-in-middle: middle part |
| `<|fim_suffix|>` | 100260 | Fill-in-middle: suffix part |
| `<|im_start|>` | 100264 | Chat: message start |
| `<|im_end|>` | 100265 | Chat: message end |

---

### Example: Document Separation

```python
# Training data with multiple documents
text = """Document 1 content here.<|endoftext|>Document 2 content here.<|endoftext|>Document 3 content here."""

token_ids = text_to_token_ids(text, tokenizer)
print(token_ids)
```

**Visual Structure:**

```
[Doc1 tokens...] [50256] [Doc2 tokens...] [50256] [Doc3 tokens...]
                    ↑                         ↑
              Document boundaries marked by special token
```

**Why this matters:**

During training, the model learns that `<|endoftext|>` means:
- Previous context is no longer relevant
- Start fresh with new document
- Don't predict across document boundaries

---

### Example: Without `allowed_special` (Error)

```python
text = "Hello<|endoftext|>World"

# ❌ This will raise an error
try:
    token_ids = tokenizer.encode(text)
except Exception as e:
    print(f"Error: {e}")
    # Output: Error: Encountered special token '<|endoftext|>' which is not allowed
```

---

### Example: With `allowed_special` (Success)

```python
text = "Hello<|endoftext|>World"

# ✅ This works
token_ids = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
print(token_ids)
# Output: [15496, 50256, 10603]
#           ↑      ↑       ↑
#        "Hello" <|eos|> "World"
```

---

## Batch Dimension Deep Dive

### Why Do Models Need Batch Dimensions?

Neural networks process data in **batches** for computational efficiency. GPUs excel at parallel operations on multiple inputs simultaneously.

**Single vs Batch Processing:**

```
Single Input (inefficient):
GPU: [Process sample 1] [Process sample 2] [Process sample 3] ...
     Sequential, underutilizes GPU

Batch Processing (efficient):
GPU: [Process samples 1,2,3,4,5,6,7,8 simultaneously]
     Parallel, maximizes GPU utilization
```

---

### Tensor Shapes at Different Stages

#### Input Text

```python
text = "Hello, world!"
# Just a string, no shape
```

#### After Encoding

```python
encoded = tokenizer.encode(text)
# [15496, 11, 995, 0]
# Python list, no PyTorch shape
```

#### After `torch.tensor()`

```python
encoded_tensor = torch.tensor(encoded)
# tensor([15496, 11, 995, 0])
# Shape: torch.Size([4])
# This is 1D: (sequence_length,)
```

#### After `unsqueeze(0)`

```python
batched_tensor = encoded_tensor.unsqueeze(0)
# tensor([[15496, 11, 995, 0]])
# Shape: torch.Size([1, 4])
# This is 2D: (batch_size, sequence_length)
```

---

### Multiple Samples (Real Batch)

```python
# Three different texts
texts = [
    "Hello!",
    "How are you?",
    "Nice to meet you."
]

# Encode each
encoded_list = [tokenizer.encode(t) for t in texts]
print(encoded_list)
# [[15496, 0], [2437, 389, 345, 30], [Nice, 284, 1826, 345, 13]]

# Pad to same length (required for batching)
max_len = max(len(e) for e in encoded_list)
padded = [e + [0] * (max_len - len(e)) for e in encoded_list]

# Create batch
batch = torch.tensor(padded)
print(batch.shape)
# torch.Size([3, 6])  ← 3 samples, 6 tokens each (padded)
```

**Visual Batch:**

```
Batch shape: (3, 6)
             ↓  ↓
         3 samples, 6 tokens each

Sample 0: [15496,    0,    0,    0,    0,    0]  "Hello!" + padding
Sample 1: [ 2437,  389,  345,   30,    0,    0]  "How are you?" + padding
Sample 2: [ 6Nice,  284, 1826,  345,   13,    0]  "Nice to meet you." + padding
           ↑                                   ↑
         Token positions 0-5
```

---

### Squeeze/Unsqueeze Operations

#### `unsqueeze(dim)` - Add Dimension

```python
# Start with 1D
x = torch.tensor([1, 2, 3, 4])
print(x.shape)  # torch.Size([4])

# Add dimension at position 0
x0 = x.unsqueeze(0)
print(x0.shape)  # torch.Size([1, 4])
print(x0)        # tensor([[1, 2, 3, 4]])

# Add dimension at position 1
x1 = x.unsqueeze(1)
print(x1.shape)  # torch.Size([4, 1])
print(x1)
# tensor([[1],
#         [2],
#         [3],
#         [4]])
```

**Visual `unsqueeze` Effects:**

```
Original (4,):
[1, 2, 3, 4]

unsqueeze(0) → (1, 4):
[[1, 2, 3, 4]]  ← Added outer brackets (batch dimension)

unsqueeze(1) → (4, 1):
[[1],
 [2],
 [3],
 [4]]  ← Added inner brackets (feature dimension)
```

#### `squeeze(dim)` - Remove Dimension

```python
# Start with 2D (size 1 in dimension 0)
x = torch.tensor([[1, 2, 3, 4]])
print(x.shape)  # torch.Size([1, 4])

# Remove dimension 0
x_squeezed = x.squeeze(0)
print(x_squeezed.shape)  # torch.Size([4])
print(x_squeezed)        # tensor([1, 2, 3, 4])
```

**Visual `squeeze` Effects:**

```
Original (1, 4):
[[1, 2, 3, 4]]

squeeze(0) → (4,):
[1, 2, 3, 4]  ← Removed outer brackets
```

---

## Common Use Cases

### Use Case 1: Interactive Text Generation

```python
def interactive_generation():
    tokenizer = tiktoken.get_encoding("gpt2")
    model.eval()
    
    while True:
        prompt = input("Enter prompt (or 'quit'): ")
        if prompt.lower() == 'quit':
            break
        
        # Convert text to tokens
        input_ids = text_to_token_ids(prompt, tokenizer)
        
        # Generate
        output_ids = generate_text_simple(
            model=model,
            idx=input_ids,
            max_new_tokens=30,
            context_size=256
        )
        
        # Convert back to text
        generated = token_ids_to_text(output_ids, tokenizer)
        print(f"Generated: {generated}\n")
```

---

### Use Case 2: Dataset Preprocessing

```python
def preprocess_dataset(texts, tokenizer, max_length=256):
    """Tokenize a list of texts for training."""
    all_token_ids = []
    
    for text in texts:
        # Tokenize
        token_ids = text_to_token_ids(text, tokenizer)
        
        # Crop if too long
        if token_ids.shape[1] > max_length:
            token_ids = token_ids[:, :max_length]
        
        all_token_ids.append(token_ids)
    
    # Stack into batch
    batch = torch.cat(all_token_ids, dim=0)
    return batch
```

---

### Use Case 3: Model Output Analysis

```python
def analyze_tokens(text, tokenizer):
    """Show detailed token breakdown."""
    token_ids = text_to_token_ids(text, tokenizer)
    
    print(f"Original text: {text}")
    print(f"Token count: {token_ids.shape[1]}")
    print(f"\nToken breakdown:")
    
    # Decode each token individually
    for i, token_id in enumerate(token_ids[0]):
        token_text = tokenizer.decode([token_id.item()])
        print(f"  Position {i}: ID {token_id:5d} → '{token_text}'")
```

**Example Output:**

```
Original text: Hello, world!
Token count: 4

Token breakdown:
  Position 0: ID 15496 → 'Hello'
  Position 1: ID    11 → ','
  Position 2: ID   995 → ' world'
  Position 3: ID     0 → '!'
```

---

### Use Case 4: Training Loop

```python
def training_step(model, text_batch, tokenizer):
    """Process a batch of texts for training."""
    # Convert all texts to token IDs
    input_ids_list = [text_to_token_ids(text, tokenizer) for text in text_batch]
    
    # Stack into single batch tensor
    input_ids = torch.cat(input_ids_list, dim=0)
    # Shape: (batch_size, seq_len)
    
    # Forward pass
    logits = model(input_ids)
    
    # Compute loss, backprop, etc.
    # ...
    
    return loss
```

---

## Troubleshooting

### Problem 1: Shape Mismatch Error

**Error:**
```
RuntimeError: Expected input batch_size (4) to match target batch_size (1)
```

**Cause:** Forgot to add batch dimension

**Solution:**
```python
# ❌ Wrong (no batch dimension)
token_ids = torch.tensor([15496, 11, 995, 0])
model(token_ids)  # Error!

# ✅ Correct (with batch dimension)
token_ids = torch.tensor([[15496, 11, 995, 0]])
model(token_ids)  # Works!

# Or use the function
token_ids = text_to_token_ids("Hello, world!", tokenizer)
model(token_ids)  # Works!
```

---

### Problem 2: Special Token Error

**Error:**
```
ValueError: Encountered special token '<|endoftext|>' which is not allowed
```

**Cause:** Didn't specify `allowed_special`

**Solution:**
```python
# ❌ Wrong
token_ids = tokenizer.encode("Text<|endoftext|>More")

# ✅ Correct
token_ids = tokenizer.encode(
    "Text<|endoftext|>More",
    allowed_special={'<|endoftext|>'}
)
```

---

### Problem 3: Can't Decode Tensor

**Error:**
```
TypeError: decode() argument must be a list, not Tensor
```

**Cause:** Forgot to convert tensor to list

**Solution:**
```python
# ❌ Wrong
token_ids = torch.tensor([15496, 11, 995, 0])
text = tokenizer.decode(token_ids)  # Error!

# ✅ Correct
text = tokenizer.decode(token_ids.tolist())

# Or use the function
text = token_ids_to_text(token_ids.unsqueeze(0), tokenizer)
```

---

### Problem 4: Wrong Batch Dimension After Squeeze

**Error:**
```
RuntimeError: squeeze() dimension 0 has size 5, but expected size 1
```

**Cause:** Trying to squeeze a dimension that's not size 1

**Solution:**
```python
# Understand your tensor shape first
token_ids = torch.tensor([[1, 2], [3, 4], [5, 6]])
print(token_ids.shape)  # torch.Size([3, 2])

# ❌ Wrong (dim 0 has size 3, not 1)
flat = token_ids.squeeze(0)  # Doesn't change anything

# ✅ Correct (only for single batch)
single_batch = torch.tensor([[1, 2, 3]])
flat = single_batch.squeeze(0)  # Works! Now shape is (3,)
```

---

## Summary

### Key Takeaways

1. **`text_to_token_ids()`**: Converts text → tensor with batch dimension
   - Encodes text to list of IDs
   - Converts to PyTorch tensor
   - Adds batch dimension with `unsqueeze(0)`

2. **`token_ids_to_text()`**: Converts tensor → text (inverse operation)
   - Removes batch dimension with `squeeze(0)`
   - Converts tensor to Python list
   - Decodes IDs back to text

3. **Special tokens** require `allowed_special` parameter to avoid errors

4. **Batch dimension** is required for all model inputs (even single samples)

5. **tiktoken** provides fast, accurate tokenization matching OpenAI models

### Function Signatures Summary

```python
def text_to_token_ids(text, tokenizer):
    """
    str → torch.Tensor
    
    Input:  "Hello, world!"
    Output: tensor([[15496, 11, 995, 0]])  # Shape: (1, 4)
    """
    
def token_ids_to_text(token_ids, tokenizer):
    """
    torch.Tensor → str
    
    Input:  tensor([[15496, 11, 995, 0]])  # Shape: (1, 4)
    Output: "Hello, world!"
    """
```

### Complete Workflow

```
    Human-Readable Text
            ↓
    [text_to_token_ids]
            ↓
    PyTorch Tensor (batch, seq_len)
            ↓
        [Model]
            ↓
    PyTorch Tensor (batch, seq_len)
            ↓
    [token_ids_to_text]
            ↓
    Human-Readable Text
```

These two simple utility functions are essential building blocks for working with GPT models, handling all the tedious conversions so you can focus on the interesting parts like generation and analysis!
