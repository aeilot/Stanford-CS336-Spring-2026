# Written Assignments

This file includes the answers to the written assignments.

## BPE

### unicode1

Unicode is a text encoding standard that maps characters to integer code points. As of Unicode 17.0 (released in September 2025), the standard defines 159,801 characters across 172 scripts.

The function `ord` and `chr`.

1. What Unicode character does chr(0) return?

'\x00' or NULL

2. How does this character’s string representation (__repr__()) differ from its printed representation?

The string representation shows the actual unicode escaped form, while the printed representation tries to display NULL (i.e. display nothing)

3. What happens when this character occurs in text? 

```py
>>> "Hello" + chr(0) + "World"
'Hello\x00World'
>>> print("Hello" + chr(0) + "World")
HelloWorld
```

It is omitted when printed out.

## unicode2

To train tokenizers directly on Unicode code points: the vocabulary would be prohibitively large
(around 150K items) and sparse (since many characters are quite rare). 

We’ll use a Unicode encoding, which converts a Unicode character into a sequence of bytes: UTF-8, UTF-16, and UTF-32.

When using byte-level tokenization, we do not need to worry about out-of-vocabulary tokens, since we know that any input text can be expressed as a sequence of integers from 0 to 255.

1. What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? 

First, UTF-8 is the dominant encoding for the Internet (more than 98% of all webpages). Using UTF-8 avoids unnecessary transcoding and compatibility issues during data ingestion.

Second, UTF-8 is more space-efficient for typical NLP corpora, especially English-heavy or ASCII-heavy text.

2. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.

"你好世界". The function is incorrect because UTF-8 characters may consist of multiple bytes. It incorrectly assumes that each byte corresponds to one complete UTF-8 character. For multibyte characters, decoding bytes individually either raises decoding errors or produces invalid results.

3. Give a two-byte sequence that does not decode to any Unicode character(s).

```text
b'\xC0\xAF'
```

This byte sequence is invalid UTF-8 because it is an overlong encoding, which UTF-8 explicitly forbids.

## BPE and Subword Tokenization

A subword tokenizer trades off a larger vocabulary size for better compression of the input byte sequence.

Byte-pair encoding is a compression algorithm that iteratively replaces (“merges”) the most frequent pair of bytes with a single, new unused index. 

### Training on Tiny Stories

```txt
Time elapsed: 114.03 seconds
Current memory usage: 41.52 MB
Peak memory usage: 51.93 MB
```

MacBook Air (M4) with 24GB RAM, multiprocessing with process_num = 8.

b' accomplishment' is the longest token.

The counter part is the longest part, according to the profiler.

### Training on OpenWebText

Due to limited computational resources, currently `TODO`.


## Transformer Language Model Architecture

A language model takes as input a batched sequence of integer token IDs (i.e., torch.Tensor of shape (batch_size, sequence_length)), and returns a (batched) normalized probability distribution over the vocabulary (i.e., a PyTorch Tensor of shape (batch_size, sequence_length, vocab_size)), where the predicted distribution is over the next word for each input token.

**Token Embeddings** Each embedding layer takes in a tensor of integers of shape (batch_size, sequence_length) and produces a sequence of vectors of shape (batch_size, sequence_length, d_model).

**Einops** The two key ops are `einsum`, which can do tensor contractions with arbitrary dimensions of input tensors, and `rearrange`, which can reorder, concatenate, and split arbitrary dimensions.

**Row-Major** Many machine learning papers use row vectors in their notation, which result in representations that mesh well with the row-major memory ordering.

**SwiGLU** Modern LLMs use SwiGLU, which combines the Swish activation function with the Gated Linear Unit (GLU) to improve model expressiveness and performance. The SiLU activation function is similar to the ReLU activation function, but is smooth at zero. (x times Sigmoid) Gated Linear Units are suggested to “reduce the vanishing gradient problem for deep architectures by providing a linear path for the gradients while retaining non-linear capabilities.”

$$FFN(x)=W_2\left(\mathrm{SiLU}(W_1x)\odot(W_3x)\right)$$, where $W_1,W_3\in\mathbb{R}^{d_{ff}\times d_{model}}$, $W_2\in\mathbb{R}^{d_{model}\times d_{ff}}$, and $\odot$ denotes element-wise multiplication.

Canonically, $d_{ff}=\frac{8}{3}d_{model}$; in practice, $d_{ff}$ is often rounded to a nearby multiple of 64 for hardware efficiency. This is basically `up projection` + `gate projection` then `down projection`.

SwiGLU combines the SiLU (Swish) activation with GLU gating, and empirical results show it outperforms standard ReLU and plain SiLU in language modeling tasks.

**RoPE** To inject positional information, RoPE rotates every pair of embedding dimensions instead of adding positional embeddings. For a query token $q^{(i)} = W_q x^{(i)} \in \mathbb{R}^d$ at position $i$, the rotated query is computed as $q'^{(i)} = R^i q^{(i)} = R^i W_q x^{(i)}$, where each pair $(q_{2k-1}, q_{2k})$ is treated as a 2D vector and rotated by the angle $\theta_{i,k} = \frac{i}{\Theta^{(2k-2)/d}}$. The corresponding rotation block is

$$
R_k^i =
\begin{pmatrix}
\cos(\theta_{i,k}) & -\sin(\theta_{i,k}) \\
\sin(\theta_{i,k}) & \cos(\theta_{i,k})
\end{pmatrix},
$$

and the full rotation matrix is a block-diagonal matrix

$$
R^i =
\begin{pmatrix}
R_1^i & 0 & \cdots & 0 \\
0 & R_2^i & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & R_{d/2}^i
\end{pmatrix},
$$

where each $0$ denotes a $2 \times 2$ zero matrix. In practice, the full $d \times d$ matrix is never constructed; instead, the precomputed values of $\cos(\theta_{i,k})$ and $\sin(\theta_{i,k})$ are cached (e.g., using `self.register_buffer(persistent=False)`) and reused across layers and batches. The same rotation is also applied to the key vectors, i.e., $k'^{(j)} = R^j k^{(j)}$, and the RoPE layer contains no learnable parameters.
