import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.utils import LlamaRotaryEmbedding, repeat_kv


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


class FeedForward(nn.Module):
    """Feed-forward module with SiLU activation."""
    def __init__(self, config):
        super(FeedForward, self).__init__()
        # Gate and up-projections project from hidden_size to intermediate_size
        self.gate_proj = nn.Linear(config.emb_dim, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.emb_dim, config.intermediate_size, bias=False)
        
        # Down projection brings the dimension back to hidden_size
        self.down_proj = nn.Linear(config.intermediate_size, config.emb_dim, bias=False)
        
        # SiLU activation function (also known as Swish)
        self.act_fn = F.silu

    def forward(self, x):
        # Apply gate and up projections
        gate_output = self.act_fn(self.gate_proj(x))  # SiLU activation
        up_output = self.up_proj(x)
        
        # Element-wise multiplication of gate and up projections
        intermediate_output = gate_output * up_output
        
        # Project back to hidden size
        output = self.down_proj(intermediate_output)
        
        return output
        