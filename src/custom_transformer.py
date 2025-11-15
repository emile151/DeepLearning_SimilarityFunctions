import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ----------------------------
# Attention function
# ----------------------------
def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

# ----------------------------
# Multi-Head Attention
# ----------------------------
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, attention_fn, dropout=0.1):
        super().__init__()
        assert d_model % h == 0, "Embedding dim must be divisible by number of heads"
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.attention_fn = attention_fn

    def forward(self, query, key, value, mask=None):
        B = query.size(0)

        if mask is not None:
            # mask is already (B,1,1,T)
            # do NOT unsqueeze again
            pass


        # Linear projections and reshape to (B, h, T, d_k)
        query, key, value = [
            l(x).view(B, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        # Apply attention
        x, self.attn = self.attention_fn(query, key, value, mask=mask, dropout=self.dropout)

        # Concatenate heads
        x = x.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.linears[-1](x)

# ----------------------------
# Small Transformer
# ----------------------------
class SmallTransformer(nn.Module):
    def __init__(self, vocab_size, attention_fn, embed_dim=24, num_heads=4, depth=2, max_len=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.01)
        self.blocks = nn.ModuleList([
            MultiHeadedAttention(num_heads, embed_dim, attention_fn)
            for _ in range(depth)
        ])
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, tokens, mask=None):
        B, T = tokens.shape
        x = self.embed(tokens) + self.pos_emb[:, :T, :]

        for blk in self.blocks:
            x = blk(x, x, x, mask=mask)

        x = self.ln(x)
        return self.head(x)

# ----------------------------
# Testing the model
# ----------------------------
def main():
    VOCAB_SIZE = 27  # 0-25 normal tokens, 26 = MASK
    MASK_TOKEN = 26
    EMBED_DIM = 128   # divisible by NUM_HEADS
    NUM_HEADS = 4
    LAYER = 4
    MAX_SEQ_LENGTH = 64

    model = SmallTransformer(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        depth=LAYER,
        max_len=MAX_SEQ_LENGTH,
        attention_fn=attention
    )

    # Example input
    test_tokens = torch.randint(0, 25, (8, 32))  # shape (B=2, T=32)

    # Pad to MAX_SEQ_LENGTH
    num_missing = MAX_SEQ_LENGTH - test_tokens.shape[1]
    if num_missing > 0:
        pad_tensor = torch.full((test_tokens.shape[0], num_missing), MASK_TOKEN, dtype=torch.long)
        test_tokens = torch.cat([test_tokens, pad_tensor], dim=1)

    # Mask: 1 for real tokens, 0 for padding
    mask = (test_tokens != MASK_TOKEN).float()  # (B,T)
    mask = mask.unsqueeze(1).unsqueeze(2)

    print("Tokens shape:", test_tokens.shape)
    print("Mask shape:", mask.shape)

    logits = model(test_tokens, mask=mask)
    print("Logits shape:", logits.shape)  # (B, T, VOCAB_SIZE)

if __name__ == "__main__":
    main()
