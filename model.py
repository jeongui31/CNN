import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import build_embeddings

class SentenceCNN(nn.Module):
    def __init__(self, mode, n_labels, word2idx, pretrained,
                 dim, num_feature_maps, filter_sizes, p_dropout, padding_idx=0):
        super().__init__()
        self.mode = mode
        self._debug_printed = False
        
        if mode == "multi":
            W = build_embeddings(word2idx, pretrained)
            W2 = W.copy()
            self.emb_static = nn.Embedding.from_pretrained(
                torch.tensor(W, dtype=torch.float32), freeze=True, padding_idx=padding_idx
            )
            self.emb_nonstatic = nn.Embedding.from_pretrained(
                torch.tensor(W2, dtype=torch.float32), freeze=False, padding_idx=padding_idx
            )
            in_ch = 2
        else:
            if mode == "rand":
                W = build_embeddings(word2idx, None)
                freeze = False
            elif mode == "static":
                W = build_embeddings(word2idx, pretrained)
                freeze = True
            elif mode == "non-static":
                W = build_embeddings(word2idx, pretrained)
                freeze = False
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            self.emb = nn.Embedding.from_pretrained(
                torch.tensor(W, dtype=torch.float32), freeze=freeze, padding_idx=padding_idx
            )
            in_ch = 1

        self.convs = nn.ModuleList([
            nn.Conv2d(in_ch, num_feature_maps, (fs, dim)) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(p=p_dropout)
        self.fc = nn.Linear(num_feature_maps * len(filter_sizes), n_labels)


    def forward(self, x):
        if not self._debug_printed:
            print("[x]", x.shape, x.dtype)  # (B, L)

        if self.mode == "multi":
            e1 = self.emb_static(x)   # (B, L, dim)
            e2 = self.emb_nonstatic(x)
            e = torch.stack([e1, e2], dim=1)  # (B, 2, L, dim)
            if not self._debug_printed:
                print("[e1,e2]", e1.shape, e2.shape)
                print("[stacked e]", e.shape)
        else:
            e = self.emb(x).unsqueeze(1)  # (B, 1, L, dim)
            if not self._debug_printed:
                print("[e]", e.shape)

        pooled = []
        for i, conv in enumerate(self.convs):
            h = F.relu(conv(e)).squeeze(3)  # (B, C, L')
            if not self._debug_printed:
                print(f"[conv{i}] kernel={conv.kernel_size} out={h.shape}")
            h = F.max_pool1d(h, h.size(2)).squeeze(2)  # (B, C)
            if not self._debug_printed:
                print(f"[pool{i}] {h.shape}")
            pooled.append(h)

        z = torch.cat(pooled, dim=1)  # (B, C * len(filter_sizes))
        if not self._debug_printed:
            print("[concat z]", z.shape)
        z = self.dropout(z)
        logits = self.fc(z)           # (B, n_labels)
        if not self._debug_printed:
            print("[logits]", logits.shape)
            self._debug_printed = True
        return logits

'''
    def forward(self, x):
        #* x : (batch_size, max_sentence_len)
        #* C : num_feature_maps 
        #* L' : L - filter_size + 1
        if self.mode == "multi":
            e1 = self.emb_static(x) #(B, L, dim)
            e2 = self.emb_nonstatic(x) #(B, L, dim)
            e = torch.stack([e1, e2], dim=1) #(B, 2, L, dim)
        else:
            e = self.emb(x).unsqueeze(1) #(B, 1, L, dim)

        pooled = []
        for conv in self.convs: #conv(e) : (B, C, L', 1)
            h = F.relu(conv(e)).squeeze(3) #(B, C, L')
            h = F.max_pool1d(h, h.size(2)).squeeze(2) #(B, C)
            pooled.append(h)

        z = torch.cat(pooled, dim=1) #(B, C * len(filter_sizes))
        z = self.dropout(z)
        logits = self.fc(z)
        return logits
'''