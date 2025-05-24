import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianRBF(nn.Module):
    """Radial Basis Function layer for distance encoding"""

    def __init__(self, K=32, cutoff=5.0):
        super().__init__()
        centers = torch.linspace(0, cutoff, K)
        self.gamma = (centers[1] - centers[0]).item() ** -2
        self.register_buffer('centers', centers)

    def forward(self, d):
        d = torch.clamp(d, min=0.0, max=10.0)
        return torch.exp(-self.gamma * (d.unsqueeze(-1) - self.centers) ** 2)


class EdgeNetwork(nn.Module):
    """Edge network for message passing"""

    def __init__(self, in_dim, emb_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, emb_dim * emb_dim)
        self.norm = nn.LayerNorm(emb_dim)
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, h, edge_index, edge_attr):
        M, E = edge_index.size(1), h.size(1)
        m = self.lin(edge_attr).view(M, E, E)
        h_j = h[edge_index[1]].unsqueeze(-1)
        m = (m @ h_j).squeeze(-1)
        aggr = torch.zeros_like(h).index_add(0, edge_index[0], m)
        return self.norm(h + aggr)


class DistanceSelfAttention(nn.Module):
    """Distance-aware self-attention mechanism"""

    def __init__(self, emb_dim, heads, dropout):
        super().__init__()
        self.h, self.d = heads, emb_dim // heads
        self.q = nn.Linear(emb_dim, emb_dim)
        self.k = nn.Linear(emb_dim, emb_dim)
        self.v = nn.Linear(emb_dim, emb_dim)
        self.out = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

        for module in [self.q, self.k, self.v, self.out]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x, dist_bias, mask):
        B, N, E = x.size()
        q = self.q(x).view(B, N, self.h, self.d).transpose(1, 2)
        k = self.k(x).view(B, N, self.h, self.d).transpose(1, 2)
        v = self.v(x).view(B, N, self.h, self.d).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / np.sqrt(self.d) + dist_bias.unsqueeze(1)
        scores = torch.clamp(scores, min=-10.0, max=10.0)

        if mask is not None:
            m = mask[:, None, None, :].bool()
            scores = scores.masked_fill(~m, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, E)
        return self.out(out)


class GraphTransformerLayer(nn.Module):
    """Graph Transformer layer with distance-aware attention"""

    def __init__(self, emb_dim, heads, dropout):
        super().__init__()
        self.attention = DistanceSelfAttention(emb_dim, heads, dropout)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x, dist_bias, mask):
        h = self.attention(x, dist_bias, mask)
        x1 = self.norm1(x + h)
        h2 = self.feedforward(x1)
        return self.norm2(x1 + h2)


class GraphEncoder(nn.Module):
    """Graph encoder with conformer ensemble and transformer layers"""

    def __init__(self, atom_dim, bond_dim, emb_dim, heads, layers, dropout, rbf_K=32, cutoff=5.0):
        super().__init__()
        self.proj = nn.Linear(atom_dim, emb_dim)
        self.rbf = GaussianRBF(rbf_K, cutoff)
        edge_input_dim = bond_dim + rbf_K
        self.edge_net = EdgeNetwork(edge_input_dim, emb_dim)
        self.layers = nn.ModuleList([
            GraphTransformerLayer(emb_dim, heads, dropout) for _ in range(layers)
        ])

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, edge_index, edge_attr, pos, batch):
        h = self.proj(x)

        # Process edge attributes
        distances = edge_attr[:, -1]
        bond_features = edge_attr[:, :-1]
        edge_features = torch.cat([bond_features, self.rbf(distances)], dim=1)
        h = self.edge_net(h, edge_index, edge_features)

        # Batch processing for transformer
        batch_size = batch.max().item() + 1
        node_lists, dist_lists, mask_lists = [], [], []

        for i in range(batch_size):
            idx = (batch == i).nonzero(as_tuple=False).squeeze()
            if idx.dim() == 0:
                idx = idx.unsqueeze(0)

            h_i, pos_i = h[idx], pos[idx]
            if pos_i.dim() == 1:
                pos_i = pos_i.unsqueeze(0)

            node_lists.append(h_i)
            mask_lists.append(torch.ones(h_i.size(0), device=h.device, dtype=torch.bool))
            dist_lists.append(torch.cdist(pos_i, pos_i))

        # Pad to same size
        max_nodes = max(nodes.size(0) for nodes in node_lists)
        x_padded = torch.stack([F.pad(nodes, (0, 0, 0, max_nodes - nodes.size(0)))
                                for nodes in node_lists])
        dist_padded = torch.stack([F.pad(dists, (0, max_nodes - dists.size(1),
                                                 0, max_nodes - dists.size(0)))
                                   for dists in dist_lists])
        mask_padded = torch.stack([F.pad(mask, (0, max_nodes - mask.size(0)))
                                   for mask in mask_lists])

        # Apply transformer layers
        for layer in self.layers:
            x_padded = layer(x_padded, dist_padded, mask_padded)

        # Global pooling
        eps = 1e-8
        mask_float = mask_padded.float().unsqueeze(-1)
        return (x_padded * mask_float).sum(1) / (mask_float.sum(1) + eps)


class SequenceEncoder(nn.Module):
    """SMILES sequence encoder using Transformer"""

    def __init__(self, vocab_size, emb_dim, heads, hidden_dim, layers, dropout):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(256, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(emb_dim, heads, hidden_dim, dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, layers)

        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)

    def forward(self, tokens):
        B, L = tokens.size()
        positions = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, L)
        x = self.token_embedding(tokens) + self.pos_embedding(positions)
        x = x.transpose(0, 1)
        return self.transformer(x).mean(0)


class FusionGating(nn.Module):
    """Gated fusion of graph and sequence representations"""

    def __init__(self, emb_dim, hidden_dim=64):
        super().__init__()
        self.gate_network = nn.Sequential(
            nn.Linear(2 * emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
            nn.Sigmoid()
        )

        for module in self.gate_network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, graph_emb, seq_emb):
        gate = self.gate_network(torch.cat([graph_emb, seq_emb], dim=-1))
        return gate * graph_emb + (1 - gate) * seq_emb


class CrossModalAttention(nn.Module):
    """Cross-modal attention between graph and sequence representations"""

    def __init__(self, emb_dim, heads, dropout, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                nn.MultiheadAttention(emb_dim, heads, dropout=dropout),
                nn.LayerNorm(emb_dim),
                nn.MultiheadAttention(emb_dim, heads, dropout=dropout),
                nn.LayerNorm(emb_dim)
            ]) for _ in range(num_layers)
        ])

    def forward(self, graph_emb, seq_emb):
        for graph_to_seq_attn, norm1, seq_to_graph_attn, norm2 in self.layers:
            g2s, _ = graph_to_seq_attn(graph_emb.unsqueeze(0), seq_emb.unsqueeze(0), seq_emb.unsqueeze(0))
            graph_emb = norm1(graph_emb + g2s.squeeze(0))

            s2g, _ = seq_to_graph_attn(seq_emb.unsqueeze(0), graph_emb.unsqueeze(0), graph_emb.unsqueeze(0))
            seq_emb = norm2(seq_emb + s2g.squeeze(0))

        return (graph_emb + seq_emb) / 2


class MultiModalRegressor(nn.Module):
    """Complete multimodal molecular property prediction model"""

    def __init__(self, atom_dim, bond_dim, vocab_size, emb_dim=128,
                 graph_heads=8, graph_layers=3, seq_heads=4, seq_layers=4,
                 dropout=0.1, graph_feat_dim=3):
        super().__init__()

        self.graph_encoder = GraphEncoder(atom_dim, bond_dim, emb_dim,
                                          graph_heads, graph_layers, dropout)
        self.sequence_encoder = SequenceEncoder(vocab_size, emb_dim, seq_heads,
                                                emb_dim * 2, seq_layers, dropout)
        self.fusion_gate = FusionGating(emb_dim)
        self.cross_modal = CrossModalAttention(emb_dim, seq_heads, dropout, layers=2)

        self.readout = nn.Sequential(
            nn.Linear(emb_dim + graph_feat_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, 1)
        )

        for module in self.readout:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, conf_batch, counts, scaffold_batch, tokens, graph_features):
        # Encode conformer graphs
        graph_emb = self.graph_encoder(conf_batch.x, conf_batch.edge_index,
                                       conf_batch.edge_attr, conf_batch.pos, conf_batch.batch)

        # Pool conformer embeddings per molecule
        conformer_embeddings = []
        start_idx = 0
        for count in counts:
            conformer_embeddings.append(graph_emb[start_idx:start_idx + count].mean(0))
            start_idx += count
        conformer_emb = torch.stack(conformer_embeddings)

        # Encode scaffold
        scaffold_emb = self.graph_encoder(scaffold_batch.x, scaffold_batch.edge_index,
                                          scaffold_batch.edge_attr, scaffold_batch.pos,
                                          scaffold_batch.batch)

        # Combine conformer and scaffold representations
        total_graph_emb = conformer_emb + scaffold_emb

        # Encode SMILES
        sequence_emb = self.sequence_encoder(tokens)

        # Fusion and cross-modal attention
        fused_emb = self.fusion_gate(total_graph_emb, sequence_emb)
        final_emb = self.cross_modal(fused_emb, fused_emb)

        # Final prediction
        combined = torch.cat([final_emb, graph_features], dim=-1)
        return self.readout(combined).squeeze(-1)

    def save_checkpoint(self, path, optimizer=None, scheduler=None):
        checkpoint = {'model': self.state_dict()}
        if optimizer:
            checkpoint['optimizer'] = optimizer.state_dict()
        if scheduler:
            checkpoint['scheduler'] = scheduler.state_dict()
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path, device='cpu', **model_kwargs):
        checkpoint = torch.load(path, map_location=device)
        model = cls(**model_kwargs).to(device)
        model.load_state_dict(checkpoint['model'])
        return model, checkpoint.get('optimizer'), checkpoint.get('scheduler')