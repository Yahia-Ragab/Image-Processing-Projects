# Transformer Attention Debugging – Image Processing Assignment

This repository contains a detailed implementation of the Transformer model (based on [“Attention Is All You Need”](https://arxiv.org/abs/1706.03762)) in PyTorch 2.8.0 with Python 3.11.9. The code is fully documented to facilitate debugging and traceability of the attention mechanism at every step.

## Features

* Full tracing of the Transformer architecture for both **encoder** and **decoder** blocks.
* Step-by-step debugging for **input embeddings**, **attention scores**, **residual connections**, **layer normalization**, and **feed-forward layers**.
* Supports inspecting **raw token IDs**, **embedding slices**, **multi-head attention splits**, and **final logits**.

## Debug Points

### Input & Embedding

1. Raw input tokens (IDs or text)
2. Target tokens (IDs or text)
3. Embedding weight matrix (slice, e.g., 5×5)
4. Input embeddings after lookup
5. Embeddings after adding positional encoding

### Encoder (Single Block, Full Trace)

6. Encoder block input tensor
7. Self-attention queries (Q)
8. Self-attention keys (K)
9. Self-attention values (V)
10. Attention score matrix **before softmax**
11. Attention score matrix **after softmax**
12. Multi-head split (Q/K/V split)
13. Multi-head attention output **after concatenation**
14. Residual connection tensors
15. Layer normalization output
16. Feed-forward input
17. Feed-forward first linear layer output
18. Feed-forward second linear layer output
19. Encoder block final output tensor

### Decoder (Single Block, Full Trace)

20. Decoder block input tensor
21. Masked self-attention queries (Q)
22. Masked self-attention keys (K)
23. Masked self-attention values (V)
24. Masked attention scores **before mask**
25. Mask tensor
26. Masked attention scores **after mask + softmax**
27. Masked self-attention multi-head split
28. Masked self-attention multi-head concatenated output
29. Residual + normalization **after masked self-attention**
30. Cross-attention queries (from decoder)
31. Cross-attention keys (from encoder)
32. Cross-attention values (from encoder)
33. Cross-attention score matrix **before softmax**
34. Cross-attention score matrix **after softmax**
35. Cross-attention output **after concatenation**
36. Residual + normalization **after cross-attention**
37. Decoder feed-forward input
38. Feed-forward first linear layer output
39. Feed-forward second linear layer output
40. Decoder block final output tensor

### Final Output

41. Decoder final sequence output (before projection)
42. Logits after final linear projection
43. Logits slice (first few values for one token)
