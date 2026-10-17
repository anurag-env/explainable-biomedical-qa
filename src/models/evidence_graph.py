"""
Evidence Graph + Graph Attention Network (GAT).

Constructs a biomedical knowledge graph G = (V, E) with:
- Nodes: Disease, Drug, Symptom, Procedure
- Edges: causes, treats, associated_with
- Aggregation via GAT: h_i' = alpha_ij * W * h_j
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """Single Graph Attention Network layer: h_i' = sum_j alpha_ij W h_j"""

    def __init__(self, in_channels, out_channels, num_heads=4, dropout=0.2,
                 add_self_loops=True, concat=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.concat = concat
        self.add_self_loops = add_self_loops

        self.W = nn.Linear(in_channels, out_channels * num_heads, bias=False)
        self.attn_src = nn.Parameter(torch.Tensor(1, num_heads, out_channels))
        self.attn_dst = nn.Parameter(torch.Tensor(1, num_heads, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels * num_heads if concat else out_channels))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index):
        """
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
        Returns:
            Updated node features [num_nodes, out_channels * num_heads] or [num_nodes, out_channels]
        """
        num_nodes = x.size(0)
        # Add self-loops
        if self.add_self_loops:
            self_loops = torch.arange(num_nodes, device=x.device).unsqueeze(0).repeat(2, 1)
            edge_index = torch.cat([edge_index, self_loops], dim=1)

        # Project: [num_nodes, num_heads, out_channels]
        h = self.W(x).view(-1, self.num_heads, self.out_channels)

        # Attention coefficients
        src, dst = edge_index[0], edge_index[1]
        alpha_src = (h[src] * self.attn_src).sum(dim=-1)  # [num_edges, num_heads]
        alpha_dst = (h[dst] * self.attn_dst).sum(dim=-1)
        alpha = self.leaky_relu(alpha_src + alpha_dst)

        # Softmax over neighbors
        alpha = self._sparse_softmax(alpha, dst, num_nodes)
        alpha = self.dropout(alpha)

        # Aggregate: h_i' = sum_j alpha_ij * h_j
        out = torch.zeros(num_nodes, self.num_heads, self.out_channels, device=x.device)
        src_features = h[src] * alpha.unsqueeze(-1)
        out.scatter_add_(0, dst.unsqueeze(-1).unsqueeze(-1).expand_as(src_features), src_features)

        if self.concat:
            out = out.view(num_nodes, self.num_heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        return out + self.bias

    def _sparse_softmax(self, alpha, index, num_nodes):
        """Compute softmax over sparse neighborhoods."""
        alpha_max = torch.zeros(num_nodes, self.num_heads, device=alpha.device)
        alpha_max.scatter_reduce_(0, index.unsqueeze(-1).expand_as(alpha), alpha, reduce="amax", include_self=False)
        alpha = torch.exp(alpha - alpha_max[index])
        alpha_sum = torch.zeros(num_nodes, self.num_heads, device=alpha.device)
        alpha_sum.scatter_add_(0, index.unsqueeze(-1).expand_as(alpha), alpha)
        return alpha / (alpha_sum[index] + 1e-16)


class EvidenceGraph(nn.Module):
    """
    Evidence Graph module that constructs a knowledge graph from biomedical
    entities and aggregates evidence using GAT layers.

    Node types: Disease, Drug, Symptom, Procedure
    Edge types: causes, treats, associated_with
    """

    NODE_TYPES = ["Disease", "Drug", "Symptom", "Procedure"]
    EDGE_TYPES = ["causes", "treats", "associated_with"]

    def __init__(self, node_embedding_dim=768, hidden_channels=256,
                 num_heads=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.node_embedding_dim = node_embedding_dim
        self.hidden_channels = hidden_channels

        # Node type embeddings
        self.node_type_embedding = nn.Embedding(len(self.NODE_TYPES), node_embedding_dim)
        # Edge type embeddings
        self.edge_type_embedding = nn.Embedding(len(self.EDGE_TYPES), node_embedding_dim)

        # GAT layers
        self.gat_layers = nn.ModuleList()
        in_dim = node_embedding_dim
        for i in range(num_layers):
            out_dim = hidden_channels if i < num_layers - 1 else hidden_channels
            concat = i < num_layers - 1
            self.gat_layers.append(
                GATLayer(in_dim, out_dim, num_heads=num_heads,
                         dropout=dropout, concat=concat)
            )
            in_dim = out_dim * num_heads if concat else out_dim

        # Project back to embedding dimension
        self.output_proj = nn.Linear(in_dim, node_embedding_dim)
        self.layer_norm = nn.LayerNorm(node_embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features, edge_index, node_types=None, edge_types=None):
        """
        Args:
            node_features: [num_nodes, node_embedding_dim]
            edge_index: [2, num_edges]
            node_types: [num_nodes] (int indices into NODE_TYPES)
            edge_types: [num_edges] (int indices into EDGE_TYPES)
        Returns:
            dict with graph_embedding and updated node_features
        """
        x = node_features

        # Add node type embeddings if provided
        if node_types is not None:
            x = x + self.node_type_embedding(node_types)

        # Apply GAT layers
        for i, gat_layer in enumerate(self.gat_layers):
            x = gat_layer(x, edge_index)
            if i < len(self.gat_layers) - 1:
                x = F.elu(x)
                x = self.dropout(x)

        # Project back and normalize
        x = self.output_proj(x)
        x = self.layer_norm(x)

        # Graph-level embedding via mean pooling
        graph_embedding = x.mean(dim=0, keepdim=True)

        return {
            "node_features": x,
            "graph_embedding": graph_embedding,
        }

    def build_graph_from_entities(self, entities, embeddings):
        """
        Build a graph from extracted biomedical entities.

        Args:
            entities: List of dicts with 'text', 'type', 'embedding'
            embeddings: Entity embeddings [num_entities, dim]

        Returns:
            node_features, edge_index, node_types
        """
        if not entities:
            dummy = torch.zeros(1, self.node_embedding_dim)
            return dummy, torch.zeros(2, 0, dtype=torch.long), torch.zeros(1, dtype=torch.long)

        node_features = embeddings
        type_map = {t: i for i, t in enumerate(self.NODE_TYPES)}
        node_types = torch.tensor(
            [type_map.get(e.get("type", "Disease"), 0) for e in entities],
            dtype=torch.long,
        )

        # Build edges: connect entities that co-occur or are related
        edges_src, edges_dst = [], []
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                if entities[i].get("type") != entities[j].get("type"):
                    edges_src.extend([i, j])
                    edges_dst.extend([j, i])

        if edges_src:
            edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)

        return node_features, edge_index, node_types
