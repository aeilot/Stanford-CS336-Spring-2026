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

