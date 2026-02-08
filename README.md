# Implementing Transformer Models with FlashAttention

This project presents an end-to-end implementation of a Transformer model for neural machine translation from German to English. The work focuses on both architectural correctness and computational efficiency, with a particular emphasis on comparing standard scaled dot-product attention against the optimized **FlashAttention** algorithm.

## Overview

The Transformer model is implemented from scratch following the architecture proposed by Vaswani et al. (2017). The project covers all major components of the model, including tokenization, positional encoding, attention mechanisms, feed-forward networks, and training strategies. In addition, FlashAttention is integrated and evaluated as a drop-in replacement for standard attention to assess its impact on runtime and memory usage.

## Experiments

The model is trained on a subset of the WMT17 German–English dataset. Performance is evaluated using BLEU score and loss curves. To assess efficiency, training speed, total runtime, and GPU memory usage are measured for both attention implementations on an NVIDIA A100 GPU.

## References

- Vaswani et al., *Attention Is All You Need*, 2017  
- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention*, 2022

## Author

Modar Alkanj
