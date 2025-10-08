# Vanilla Vision Transformer (ViT)

This repository implements a **vanilla Vision Transformer (ViT)** based on the paper [Unleashing Vanilla Vision Transformer with Masked Image Modeling for Object Recognition](https://openaccess.thecvf.com/content/ICCV2023/papers/Fang_Unleashing_Vanilla_Vision_Transformer_with_Masked_Image_Modeling_for_Object_ICCV_2023_paper.pdf) (ICCV 2023), using **Python 3.11.8** and **PyTorch 2.2.0**.

## Overview

### Input and Patchification
- Raw input image tensor (after preprocessing)  
- Image divided into patches (before flattening)  
- Flattened patches (reshaped into vectors)  

### Embedding and Tokens
- Patch embeddings after linear projection  
- Class token before concatenation  
- Embeddings after adding the class token  
- Embeddings after adding positional encoding  

### Encoder Block (Trace One Block)
- Encoder block input tensor  
- Multi-head attention queries (Q)  
- Multi-head attention keys (K)  
- Multi-head attention values (V)  
- Attention scores before softmax  
- Attention scores after softmax  
- Multi-head attention output (after concatenation)  
- Residual connection + normalization (post-attention)  
- Feed-forward input  
- Feed-forward hidden layer output  
- Feed-forward output after second linear  
- Residual connection + normalization (post-MLP)  
- Encoder block final output  

### Deeper Encoder Blocks
- Encoder block 2 output  
- Encoder block N (last block) output  

### Final Output
- Final sequence output (including class token)  
- Class token extracted (final representation)  
- Classification head logits  
- Softmax probabilities (example slice)  
