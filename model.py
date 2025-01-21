import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int = 64,               # Dimension per attention head
        max_seq_len: int = 2048,     # Maximum sequence length
        base: int = 10000,           # Base for the angle calculations
        device: str = None
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Create cache for position frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Create position sequence
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        
    def _update_cos_sin_tables(self, x: torch.Tensor, seq_len: int):
        # Return early if cache is valid
        if seq_len <= self._seq_len_cached:
            return
        
        # Update cache size
        self._seq_len_cached = seq_len
        
        # Create position sequence
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        # Calculate position frequencies
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        
        # Calculate embeddings
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos_cached = emb.cos()[None, None, :, :]
        self._sin_cached = emb.sin()[None, None, :, :]
        
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, num_heads, head_dim = q.shape
        
        # Update cos/sin tables if needed
        self._update_cos_sin_tables(q, seq_len)
        
        # Get cos and sin for current sequence
        cos = self._cos_cached[:, :, :seq_len, :]
        sin = self._sin_cached[:, :, :seq_len, :]
        
        def rotate_half(x):
            """Rotates half the hidden dims of the input."""
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)
        
        # Apply rotary embeddings to q and k
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        
        return q_embed, k_embed
    

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Attention(nn.Module):
    """Multi-head attention module with support for GQA (Grouped Query Attention)."""
    def __init__(self, config):
        super(Attention, self).__init__()
        self.emb_dim = config.emb_dim
        self.n_q_heads = config.n_q_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = self.emb_dim // self.n_q_heads
        self.n_rep = self.n_q_heads // self.n_kv_heads

        # Projections for Q, K, V & O
        self.q_proj = nn.Linear(self.emb_dim, self.emb_dim)
        self.k_proj = nn.Linear(self.emb_dim, self.head_dim * self.n_kv_heads)
        self.v_proj = nn.Linear(self.emb_dim, self.head_dim * self.n_kv_heads)
        self.o_proj = nn.Linear(self.emb_dim, self.emb_dim)
        
        # Initialize rotary embeddings
        self.rotary_embedding = LlamaRotaryEmbedding(dim=self.head_dim)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)).view(
                1, 1, config.max_seq_len, config.max_seq_len
            ),
        )

    def forward(self, x):
        B, T, C = x.size()  # batch_size, seq_len, emb_dim

        # Project Q, K, V
        q = self.q_proj(x)  # (B, T, emb_dim)
        k = self.k_proj(x)  # (B, T, n_kv_heads * head_dim)
        v = self.v_proj(x)  # (B, T, n_kv_heads * head_dim)

        # Reshape Q, K, V
        q = q.view(B, T, self.n_q_heads, self.head_dim)  # (B, T, n_q_heads, head_dim)
        k = k.view(B, T, self.n_kv_heads, self.head_dim)  # (B, T, n_kv_heads, head_dim)
        v = v.view(B, T, self.n_kv_heads, self.head_dim)  # (B, T, n_kv_heads, head_dim)

        # Apply rotary embeddings
        q, k = self.rotary_embedding(q, k)
        
        # Reshape for attention computation
        q = q.transpose(1, 2)  # (B, n_q_heads, T, head_dim)
        k = k.transpose(1, 2)  # (B, n_kv_heads, T, head_dim)
        v = v.transpose(1, 2)  # (B, n_kv_heads, T, head_dim)

        # Repeat K and V for GQA
        k = repeat_kv(k, self.n_rep)  # (B, n_q_heads, T, head_dim)
        v = repeat_kv(v, self.n_rep)  # (B, n_q_heads, T, head_dim)

        # Compute attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        att = (q @ k.transpose(-2, -1)) * scale  # (B, n_q_heads, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Apply attention to values
        y = att @ v  # (B, n_q_heads, T, head_dim)

        # Reshape and project output
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, emb_dim)
        y = self.o_proj(y)
        y = self.resid_dropout(y)

        return y

        

        