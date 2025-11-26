import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ----------------------------
# Multi-Head Attention
# ----------------------------
class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_fn=None, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "Embedding dim must be divisible by number of heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Linear projections
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

        # Custom Attention function

        if attn_fn == 'rbf':
            self.gamma = nn.Parameter(torch.tensor(0.2))
            self.attn_fn = self.rbf_attention
        else:
            self.attn_fn = self.dot_prod_attention

    def dot_prod_attention(self, query, key, value, mask , dropout=None):
        d_k = query.size(-1)
        ##### To be changed in each attention function
        scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)
        if mask is not None:
            # Expand mask to match scores
            mask = mask[:, None, None, :]  # [B, 1, 1, T]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        #####

        out = attn_weights @ value
        return out
    
    def rbf_attention(self, query, key, value, mask , dropout=None):
        d_k = query.size(-1)
        q_exp = query.unsqueeze(3)   # (B, H, Tq, 1, d_k)
        k_exp = key.unsqueeze(2)     # (B, H, 1, Tk, d_k)
        ##### To be changed in each attention function
        dist = (q_exp - k_exp).pow(2).sum(-1)   # (B, H, Tq, Tk)
        rbf = torch.exp(-dist / self.gamma)   # (B, C)
        scores = rbf  / math.sqrt(d_k)
        #####

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = attn_weights @ value
        return out

    def forward(self, query, key, value, mask=None):
        B, T, D = query.shape

        Q = self.Wq(query)
        K = self.Wk(key)
        V = self.Wv(value)

        Q = Q.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.d_head).transpose(1, 2)

        # print(f"[DEBUG] query, Q is {Q} with shape {Q.shape}")
        # print(f"[DEBUG] mask is {mask} with shape {mask.shape}")
        scores = self.attn_fn(query = Q, key = K, value = V, mask = mask)

        out = scores.transpose(1, 2).reshape(B, T, D)

        out = self.Wo(out)

        return out

# ----------------------------
# Transformer Encoder Block
# ----------------------------
class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, dim_ff, attn_func, dropout=0.1):
        super().__init__()

        # 1. Multi-head self-attention block
        self.self_attn = MultiHeadedAttention(emb_dim, num_heads, attn_fn=attn_func)

        # 2. Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, emb_dim)
        )

        # 3. Normalization layer
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

        # 4. Dropout layer
        self.dropout = nn.Dropout()

    def forward(self, x, mask=None):
        # 1. Self-attention + residual + norm
        # 1.1 calculate self-attention
        attn_out = self.self_attn(x, x, x, mask)
        # 1.2 residual
        x = x + self.dropout(attn_out)
        # 1.3 normilzation
        x = self.norm1(x)

        # 2. Feed-formward + residual + norm
        # 2.1 Feed-forward
        ffn_out = self.ffn(x)
        # 2.2 residual
        x = x + self.dropout(ffn_out)
        # 2.3 normilization
        x = self.norm2(x)

        return(x)

# ----------------------------
# Small Transformer
# ----------------------------
class SmallTransformer(nn.Module):
    def __init__(self, vocab_size, attention_fn=None, embed_dim=128, num_heads=4, depth=4, max_len=128):
        super().__init__()
        # Create internal learnable representation of input data
        # embed is a lookup table for each input
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # pos_emb is a learnable matrix of positional encoding
        # matrix is random initialzed with small numbers to not overwhelm in the beginning
        scale = 0.0001
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, embed_dim) * scale)

        # Create a stack of encoder layers (depth times)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, 4 * embed_dim, attention_fn)
            for _ in range(depth)
        ])

    def forward(self, tokens, mask=None):
        B, T = tokens.shape
        x = self.embed(tokens) + self.pos_emb[:, :T, :]
 
        for blk in self.blocks:
            x = blk(x, mask=mask)

        return x
    
    def forward_block(self, tokens, idx, mask=None):
        B, T = tokens.shape
        x = self.embed(tokens) + self.pos_emb[:, :T, :]
 
        for i in range(idx + 1):
            blk = self.blocks[i]
            x = blk(x, mask=mask)

        return x
    

# ----------------------------
# Testing the model
# ----------------------------
def main():
    VOCAB_SIZE = 27  # 0-25 normal tokens, 26 = MASK
    MASK_TOKEN = 1
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

    # # Pad to MAX_SEQ_LENGTH
    # num_missing = MAX_SEQ_LENGTH - test_tokens.shape[1]
    # if num_missing > 0:
    #     pad_tensor = torch.full((test_tokens.shape[0], num_missing), MASK_TOKEN, dtype=torch.long)
    #     test_tokens = torch.cat([test_tokens, pad_tensor], dim=1)

    # # Mask: 1 for real tokens, 0 for padding
    # mask = (test_tokens != MASK_TOKEN).float()  # (B,T)
    # mask = mask.unsqueeze(1).unsqueeze(2)

    print("Tokens shape:", test_tokens.shape)
    print("Mask shape:", mask.shape)

    logits = model(test_tokens, mask=mask)
    print("Logits shape:", logits.shape)  # (B, T, VOCAB_SIZE)

if __name__ == "__main__":
    main()
