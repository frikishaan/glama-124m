import inspect
from functools import cached_property
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from typing import Optional, Tuple
import math
from dataclasses import dataclass
from torch.nn.attention.flex_attention import BlockMask, flex_attention, create_block_mask

@dataclass
class ModelConfig():
    emb_dim: int = 768
    rope_theta: int = 10000
    n_layers: int = 12
    n_heads: int = 12
    ffn_dim: int = 3072
    dropout: float = 0.1
    weight_decay: float = 1e-1
    vocab_size: int = 50257 # GPT-2 tokenizer
    eos_token_id: int = 50256
    rms_norm_eps: float = 1e-6
    dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    seq_len: int = 1024
    batch_size: int = 16

def precompute_freqs_cis(head_dim: int, seq_len: int, theta: float = 10000.0, dtype=torch.float32)-> Tuple[Tensor, Tensor]:
    # Compute the inverse frequencies: 1 / (theta ^ (2i / dim))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    
    # Generate position indices [0, 1, ..., seq_len-1]
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    
    # Outer product to get (seq_len, head_dim/2)
    freqs = torch.outer(t, inv_freq)
    
    # Precompute cos and sin. 
    # We repeat them to match head_dim so we can apply them via element-wise multiplication.
    # Shape: (seq_len, head_dim)
    emb = torch.cat((freqs, freqs), dim=-1)
    
    return emb.cos().to(dtype), emb.sin().to(dtype)

def apply_rope(x: Tensor, cos: Tensor, sin: Tensor)-> Tensor:
    """
    x: (batch, seq_len, num_heads, head_dim)
    cos, sin: (seq_len, head_dim)
    """
    # Reshape cos/sin for broadcasting: (1, seq_len, 1, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    
    # Split x into two halves
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    
    # Rotate: [-x2, x1]
    x_rotated = torch.cat((-x2, x1), dim=-1)
    
    # Standard formula: x * cos(theta) + rotate_half(x) * sin(theta)
    return (x * cos) + (x_rotated * sin)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) # gamma

    def _norm(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        # (Dim) * (B, Seq_Len, Dim) = (B, Seq_Len, Dim)
        return self.weight * self._norm(x.float()).type_as(x)

class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        assert config.emb_dim % config.n_heads == 0, "emb_dim is not divisible by n_heads"
        self.dk = config.emb_dim // config.n_heads
        self.n_head = config.n_heads

        self.wq = nn.Linear(config.emb_dim, config.emb_dim, bias=False)
        self.wk = nn.Linear(config.emb_dim, config.emb_dim, bias=False)
        self.wv = nn.Linear(config.emb_dim, config.emb_dim, bias=False)
        
        self.out_proj = nn.Linear(config.emb_dim, config.emb_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)

    def reset_kv_cache(self):
        self.cache_k  = None
        self.cache_v = None

    def forward(
        self, 
        x: Tensor, 
        freq: Tensor, 
        block_mask: BlockMask, 
        use_cache: bool = False,
        start_pos: int = 0
    ):
        batch, seq_len, _ = x.shape

        cos = freq[0][start_pos: start_pos + seq_len].to(x.device)
        sin = freq[1][start_pos: start_pos + seq_len].to(x.device)

        # (batch, seq_len, emb_dim) -> (batch, seq_len, emb_dim)
        query = self.wq(x)
        key = self.wk(x)
        value = self.wv(x)

        # (batch, seq_len, emb_dim) -> (batch, seq_len, n_head, d_k)
        query = query.view(batch, seq_len, self.n_head, self.dk)
        key = key.view(batch, seq_len, self.n_head, self.dk)
        value = value.view(batch, seq_len, self.n_head, self.dk)

        # Apply rotary embeddings
        query = apply_rope(query, cos, sin)
        key = apply_rope(key, cos, sin)

        # (batch, seq_len, n_head, d_k) -> (batch, n_head, seq_len, d_k)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if use_cache:
            if self.cache_k is None:
                # pre-allocate tensors of max seq length
                self.cache_k = torch.zeros(batch, self.n_head, self.config.seq_len, self.dk, device=x.device, dtype=key.dtype)
                self.cache_v = torch.zeros(batch, self.n_head, self.config.seq_len, self.dk, device=x.device, dtype=value.dtype)

            self.cache_k[:, :, start_pos:start_pos+seq_len] = key
            self.cache_v[:, :, start_pos:start_pos+seq_len] = value

            key = self.cache_k[:, :, :start_pos+seq_len]
            value = self.cache_v[:, :, :start_pos+seq_len]

            attn = F.scaled_dot_product_attention(
                query, key, value,
                is_causal=(seq_len > 1),
            )
        else:
            # (batch, h, seq_len, d_k) -> (batch, h, seq_len, d_k)
            attn = flex_attention(
                query, key, value,
                block_mask=block_mask
            )

            # if flex_attention not supported, fallback to SDPA
            # also need to create block mask manually
            # attn = torch.nn.functional.scaled_dot_product_attention(
            #     query, key, value, 
            #     attn_mask=None, 
            #     dropout_p=self.config.dropout if self.training else 0, 
            #     is_causal=True
            # )

        y = attn.transpose(1, 2).contiguous().view(batch, seq_len, self.n_head * self.dk)

        return self.dropout(self.out_proj(y))

class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        ffn_dim = config.ffn_dim or config.emb_dim * 4
        self.fc = nn.Linear(config.emb_dim, ffn_dim)
        self.gelu = nn.GELU(approximate='tanh')
        self.out_proj = nn.Linear(ffn_dim, config.emb_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor):
        x = self.fc(x)
        x = self.gelu(x)
        x = self.out_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.attn_norm = RMSNorm(config.emb_dim, config.rms_norm_eps)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.emb_dim, config.rms_norm_eps)
        self.ffn = FeedForward(config)

    def forward(
        self, 
        x: Tensor, 
        freq: Tensor, 
        block_mask: BlockMask,
        use_cache: bool = False,
        start_pos: int = 0
    ):
        x = x + self.attention(self.attn_norm(x), freq, block_mask, use_cache, start_pos)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class GLaMA(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.in_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.blocks = nn.ModuleList([
            Block(config) for _ in range(config.n_layers)
        ])
        self.norm = RMSNorm(config.emb_dim, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.in_emb.weight # weight tying

        # init weights
        self.apply(self._init_weights)

        # apply special scaled init to the residual projections, per GPT-2 paper
        # see - https://github.com/karpathy/nanoGPT/blob/master/model.py
        for pn, p in self.named_parameters():
            if pn.endswith('out_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers))

        cos, sin = precompute_freqs_cis(self.config.emb_dim // self.config.n_heads, self.config.seq_len, self.config.rope_theta, self.config.dtype)

        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @cached_property
    def get_num_params(self) -> int:
        n_params = sum(p.numel() if p.requires_grad else 0 for p in self.parameters())
        return n_params

    def init_optimizer(self, weight_decay: float = 0.1, beta1: float = 0.9, beta2: float = 0.95, lr: float = 6e-4, device: str = 'cpu') -> torch.optim.Optimizer:
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        # num_decay_params = sum(p.numel() for p in decay_params)
        # num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        # print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(beta1, beta2), **extra_args)
        print(f"Using fused AdamW: {use_fused}")

        return optimizer

    def init_scheduler(self, optimizer: torch.optim.Optimizer, warmup_iters: int, lr_decay_iters: int, min_lr: float, max_lr: float) -> torch.optim.lr_scheduler.LambdaLR:
        def lr_lambda(iters):
            # warmup
            if iters < warmup_iters:
                return iters / warmup_iters

            # past decay
            if iters > lr_decay_iters:
                return min_lr / max_lr

            # cosine decay
            decay_ratio = (iters - warmup_iters) / (lr_decay_iters - warmup_iters)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return (min_lr + coeff * (max_lr - min_lr)) / max_lr

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # Taken from - https://github.com/karpathy/nanoGPT/blob/master/model.py
        N = self.get_num_params
        cfg = self.config
        L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.emb_dim//cfg.n_heads, cfg.seq_len
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    def _make_doc_causal_block_mask(self, doc_ids: Tensor, seq_len: int, device: torch.device) -> BlockMask:
        # (B, S)
        B = doc_ids.size(0)

        def document_causal_mask(b, h, q_idx, kv_idx):
            # Causal: query can only attend to previous positions
            causal_mask = q_idx >= kv_idx
            # Document: tokens must be in same document
            document_mask = doc_ids[b, q_idx] == doc_ids[b, kv_idx]
            return causal_mask & document_mask

        return create_block_mask(
            document_causal_mask,
            B=B,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=device,
        )

    def forward(
        self, x: Tensor, 
        target: Tensor|None = None, 
        doc_ids: Tensor|None = None, 
        output_hidden_state: bool = False,
        use_cache: bool = False,
        start_pos: int = 0
    ) -> tuple[Tensor, Optional[Tensor]]:

        # Create doc_ids if not provided
        if doc_ids is None:
            doc_ids = torch.zeros_like(x)  # Single document

        block_mask = self._make_doc_causal_block_mask(doc_ids.to(x.device), seq_len=x.size(1), device=x.device)

        x = self.in_emb(x)
        
        for block in self.blocks:
            x = block(x, (self.cos, self.sin), block_mask, use_cache, start_pos)
        x = self.norm(x)

        # output hidden state for classification
        if output_hidden_state:
            return x

        if target is not None:
            # if targets are given, calculate loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=-1)
        else:
            # mini-optimzation for inference, only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None) -> Tensor:
        """
        Generate text by autoregressively calling the model
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.seq_len else idx[:, -self.config.seq_len:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

            if idx_next == self.config.eos_token_id:
                break

        return idx

    def reset_kv_cache(self):
        for block in self.blocks:
            block.attention.reset_kv_cache()


    @torch.no_grad()
    def generate_with_cache(
        self, 
        idx: Tensor, 
        max_new_tokens: int, 
        temperature: float = 1.0, 
        top_k: int | None = None
    ) -> Tensor:
    
        self.reset_kv_cache()
        start_pos = 0

        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= self.config.seq_len else idx[:, -self.config.seq_len:]
        logits, _ = self(idx_cond, use_cache=True, start_pos=start_pos)

        for _ in range(max_new_tokens):
            
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

            if idx_next == self.config.eos_token_id:
                break

            start_pos = idx.size(1) - 1

            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_next, use_cache=True, start_pos=start_pos)

        return idx