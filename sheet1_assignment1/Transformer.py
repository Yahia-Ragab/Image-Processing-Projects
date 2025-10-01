"""
Reduced Transformer implementation with detailed debug breakpoints.
Each important tensor step is logged with a comment label (1–43).
You can insert `print()` or actual debugger breakpoints at these marked points
for a full forward-pass trace.
"""

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, values, keys, query, mask=None, debug=False, tag=""):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.values(values)  # (7/22) Values (V)
        keys = self.keys(keys)        # (8/23) Keys (K)
        queries = self.queries(query) # (7/21/30) Queries (Q)

        # Split into heads (12/27)
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        # energy: (10/24/33) raw attention scores
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            # (25) Mask tensor
            energy = energy.masked_fill(mask == 0, float("-1e20")) # (24→26/33→34)

        attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=3) # (11/26/34)
        attention = self.dropout(attention)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # (13/28/35) multi-head concat

        out = self.fc_out(out)

        return out, queries, keys, values, energy, attention


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask=None, debug=False):
        attention_out, Q, K, V, energy, att = self.attention(value, key, query, mask)
        # (14) residual connection
        x = self.dropout(self.norm1(attention_out + query)) # (15)
        forward_in = x # (16)
        ff1 = self.feed_forward[0](forward_in) # (17)
        ff1 = self.feed_forward[1](ff1)
        ff2 = self.feed_forward[2](ff1) # (18)
        out = self.dropout(self.norm2(ff2 + x)) # (19)

        return out, (Q, K, V, energy, att, attention_out, forward_in, ff1, ff2)


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):

        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, debug=False):
        N, seq_length = x.shape
        # (1) raw input tokens
        # (3) embedding weight matrix sample
        embed_weights = self.word_embedding.weight[:5, :5]
        # (4) input embeddings after lookup
        word_emb = self.word_embedding(x)
        # (5) embeddings after adding positional
        positions = torch.arange(0, seq_length).unsqueeze(0).expand(N, seq_length).to(self.device)
        pos_emb = self.position_embedding(positions)
        out = self.dropout(word_emb + pos_emb)

        for layer in self.layers:
            out, dbg = layer(out, out, out, mask)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.attention = SelfAttention(embed_size, heads=heads, dropout=dropout)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask=None, trg_mask=None, debug=False):
        att_out, Qd, Kd, Vd, energy_d, att_d = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(att_out + x)) # (29)
        out, dbg = self.transformer_block(value, key, query, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, trg_mask=None, debug=False):
        N, seq_length = x.shape
        # (2) target tokens
        positions = torch.arange(0, seq_length).unsqueeze(0).expand(N, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions)) # (20)

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out_before_proj = x # (41)
        out = self.fc_out(x) # (42)
        out_slice = out[0,0,:5] # (43)
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=128,
        num_layers=2,
        forward_expansion=4,
        heads=4,
        dropout=0.1,
        device="cpu",
        max_length=100,
    ):

        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        subsequent_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        subsequent_mask = subsequent_mask.unsqueeze(0).unsqueeze(1)
        trg_mask = trg_pad_mask & subsequent_mask
        return trg_mask.to(self.device)

    def forward(self, src, trg, debug=False):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask, debug=debug)
        out = self.decoder(trg, enc_src, src_mask, trg_mask, debug=debug)
        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0]]).to(device)
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0]]).to(device)

    model = Transformer(50, 50, 0, 0, device=device).to(device)
    out = model(x, trg[:, :-1], debug=True)
    print("Output shape:", out.shape)
