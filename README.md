# GLaMA (Generalized Lightweight Autoregressive Model with Attention)

GLaMA is a small-scale autoregressive transformer model inspired by GPT-style architectures.
This project is built for **learning and experimentation** purposes only.

The goal of this repository is to understand the mechanics of training modern decoder-only language models from scratch, including architectural improvements and efficient training techniques.

## Model Architecture

The configuration is comparable to GPT-2 Small / LLaMA-style models:

- **Context length** - 1024
- **Embedding dimension** - 768
- **Number of attention heads** - 12
- **Number of layers** - 12
- **Data type** - BFloat16

## Architectural Modifications

I have added the following improvements to the original architecture:

- **RoPE (Rotary Positional Embeddings)** instead of learned positional embeddings to encode relative positional information and improve extrapolation to longer contexts.
- **RMSNorm** instead of LayerNorm to simplify normalization while maintaining training stability with lower computational overhead.

## Training Setup

- **Total tokens trained**: ~3B
- **Optimizer**: AdamW
- **Learning rate scheduler**: Cosine decay
- **Total training steps**: ~15K
- **Hardware**: Single A100 (40GB)
- **Training time**: ~6 hours
- **Throughput**: ~170k tokens/second
- **Estimated Model FLOPs utilization**: ~48%

FLOPs were calculated using the implementation from [NanoGPT](https://github.com/karpathy/nanoGPT/blob/master/model.py#L289).

## Dataset

Training was performed on [PrimeCorpus-1B](https://huggingface.co/datasets/frikishaan/PrimeCorpus-1B), a 1B-token high-quality dataset composed of multiple sources.

## Efficiency Improvements

This implementation uses PyTorch **FlexAttention**, allowing multiple documents to be concatenated within a sequence using a document mask combined with a causal mask.

This avoids wasted computation on padding tokens and improves overall training efficiency.

## Setup & Usage

### Check FlexAttention Support

```bash
python check_attn.py
```

This verifies whether your PyTorch installation and hardware support FlexAttention.

### Prepare dataset

To preprocess and prepare the dataset, run:

```bash
python prepare_dataset.py
```
### Start training

```bash
python train.py
```

## Results

| Metric | Value | Original |
| --- | --- | --- |
| Train loss | 2.9 | ~2.57  |
| Validation loss | 2.7 | ~3.09 |
| Perplexity | 16 | 10-20 |
| Hellaswag score | 27.5% | 29.4% |


_Note: The "Original" values are taken from community-reported benchmarks and public reproductions, not directly from the original paper. Differences may arise due to dataset, tokenizer, evaluation setup, and training details._

## Purpose of This Project

This repository is intended strictly for:

- Educational use
- Architecture experimentation
- Understanding scaling behavior
- Reproducing small LLM training pipelines

It is not intended for production deployment.

## Acknowledgements & References

This project draws inspiration from and reuses ideas from the following resources:

- [NanoGPT](https://github.com/karpathy/nanoGPT)
- [LLaMA explained - By Umar Jamil](https://www.youtube.com/watch?v=Mn_9W1nCFLo)
