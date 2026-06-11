"""
model_gat.py
============

Improved heterogeneous GAT model for drug-disease association prediction.

Main improvements:
- Proper heterogeneous GAT with relation-specific attention
- LayerNorm instead of BatchNorm (more stable for small biomedical graphs)
- Separate normalization per node type
- Residual connections
- Multi-head attention
- Layer-wise semantic attention
- Safer decoder regularization
- Stronger anti-overfitting defaults
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from dgl.nn import GATConv


# =============================================================================
# Bilinear Decoder
# =============================================================================

class InnerProductDecoder(nn.Module):

    def __init__(self, input_dim: int, dropout: float = 0.4):
        super().__init__()

        self.dropout = nn.Dropout(min(dropout, 0.5))

        self.W_drug = nn.Linear(input_dim, input_dim, bias=False)
        self.W_disease = nn.Linear(input_dim, input_dim, bias=False)

        nn.init.xavier_uniform_(self.W_drug.weight)
        nn.init.xavier_uniform_(self.W_disease.weight)

    def forward(self, feature: dict):

        drug = self.dropout(feature['drug'])
        disease = self.dropout(feature['disease'])

        drug = self.W_drug(drug)
        disease = self.W_disease(disease)

        score = drug @ disease.T

        return score


# =============================================================================
# Semantic Attention
# =============================================================================

class SemanticAttention(nn.Module):

    def __init__(self, in_feats: int, hidden_size: int = 128):
        super().__init__()

        self.project = nn.Sequential(
            nn.Linear(in_feats, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):

        w = self.project(z).mean(0)

        beta = torch.softmax(w, dim=0)

        beta = beta.expand(z.shape[0], *beta.shape)

        return (beta * z).sum(1)


# =============================================================================
# Heterogeneous GAT Layer
# =============================================================================

class HeteroGATLayer(nn.Module):

    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads,
        rel_names,
        dropout=0.4,
        residual=True,
        last_layer=False,
    ):
        super().__init__()

        self.last_layer = last_layer
        self.num_heads = num_heads
        self.out_feats = out_feats
        self.residual = residual

        conv_dict = {}

        for rel in rel_names:

            conv_dict[rel] = GATConv(
                in_feats=in_feats,
                out_feats=out_feats,
                num_heads=num_heads,
                feat_drop=dropout,
                attn_drop=dropout,
                residual=False,
                activation=None,
                allow_zero_in_degree=True,
            )

        self.conv = dglnn.HeteroGraphConv(
            conv_dict,
            aggregate='sum'
        )

        merged_dim = out_feats if last_layer else out_feats * num_heads

        if residual:
            self.res_fc = nn.Linear(in_feats, merged_dim, bias=False)
            nn.init.xavier_normal_(self.res_fc.weight)
        else:
            self.res_fc = None

        # Separate normalization per node type
        self.norm_drug = nn.LayerNorm(merged_dim)
        self.norm_disease = nn.LayerNorm(merged_dim)

        self.activation = nn.PReLU()

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        graph,
        inputs,
        apply_norm=True,
        apply_dropout=True,
    ):

        h = self.conv(graph, inputs)

        out = {}

        for ntype, feat in h.items():

            # feat: [N, heads, out_feats]

            if self.last_layer:
                feat = feat.mean(dim=1)
            else:
                feat = feat.flatten(start_dim=1)

            # Residual
            if self.residual and self.res_fc is not None:
                feat = feat + self.res_fc(inputs[ntype])

            # Separate normalization
            if apply_norm:

                if ntype == 'drug':
                    feat = self.norm_drug(feat)
                else:
                    feat = self.norm_disease(feat)

            feat = self.activation(feat)

            if apply_dropout:
                feat = self.dropout(feat)

            out[ntype] = feat

        return out


# =============================================================================
# Main GAT Model
# =============================================================================

class ModelGAT(nn.Module):

    def __init__(self, args, etypes, ntypes, in_feats):

        super().__init__()

        self.ntypes = ntypes
        self.model_type = args.concatenate_type

        hidden_feats = args.hidden_feats
        num_heads = args.num_heads
        dropout = args.dropout

        # =========================================================================
        # Input projection
        # =========================================================================

        self.drug_proj = nn.Linear(in_feats[0], hidden_feats)
        self.disease_proj = nn.Linear(in_feats[1], hidden_feats)

        nn.init.xavier_normal_(self.drug_proj.weight)
        nn.init.xavier_normal_(self.disease_proj.weight)

        # =========================================================================
        # Structural graph encoder
        # =========================================================================

        self.gat1 = HeteroGATLayer(
            in_feats=hidden_feats,
            out_feats=hidden_feats,
            num_heads=num_heads,
            rel_names=etypes,
            dropout=dropout,
            residual=True,
            last_layer=True,
        )

        self.gat2 = HeteroGATLayer(
            in_feats=hidden_feats,
            out_feats=hidden_feats,
            num_heads=num_heads,
            rel_names=etypes,
            dropout=dropout,
            residual=True,
            last_layer=True,
        )

        # =========================================================================
        # LLM branch
        # =========================================================================

        if self.model_type in ['graph_graph', 'cross_graph', 'graph_ae']:

            self.drug_proj_llm = nn.Linear(in_feats[2], hidden_feats)
            self.disease_proj_llm = nn.Linear(in_feats[3], hidden_feats)

            nn.init.xavier_normal_(self.drug_proj_llm.weight)
            nn.init.xavier_normal_(self.disease_proj_llm.weight)

        # =========================================================================
        # graph_graph / cross_graph
        # =========================================================================

        if self.model_type in ['graph_graph', 'cross_graph']:

            self.gat_llm1 = HeteroGATLayer(
                in_feats=hidden_feats,
                out_feats=hidden_feats,
                num_heads=num_heads,
                rel_names=etypes,
                dropout=dropout,
                residual=True,
                last_layer=True,
            )

            self.gat_llm2 = HeteroGATLayer(
                in_feats=hidden_feats,
                out_feats=hidden_feats,
                num_heads=num_heads,
                rel_names=etypes,
                dropout=dropout,
                residual=True,
                last_layer=True,
            )

        # =========================================================================
        # graph_ae
        # =========================================================================

        elif self.model_type == 'graph_ae':

            self.mlp_llm1 = nn.Linear(hidden_feats, hidden_feats)
            self.mlp_llm2 = nn.Linear(hidden_feats, hidden_feats)

            nn.init.xavier_normal_(self.mlp_llm1.weight)
            nn.init.xavier_normal_(self.mlp_llm2.weight)

            self.mlp_activation = nn.PReLU()

        # =========================================================================
        # Layer attention
        # =========================================================================

        self.layer_att_drug = SemanticAttention(hidden_feats)
        self.layer_att_disease = SemanticAttention(hidden_feats)

        if self.model_type not in ['none', 'as_node']:

            self.layer_att_drug_llm = SemanticAttention(hidden_feats)
            self.layer_att_disease_llm = SemanticAttention(hidden_feats)

        # =========================================================================
        # Decoder
        # =========================================================================

        self.decoder = InnerProductDecoder(
            hidden_feats,
            dropout=dropout
        )

    # =============================================================================
    # Forward
    # =============================================================================

    def forward(self, g, x):

        # =========================================================================
        # Graph unpacking
        # =========================================================================

        if isinstance(g, list):

            g_llm = g[1]
            g = g[0]

        else:

            g_llm = g

        # =========================================================================
        # Structural branch
        # =========================================================================

        h = {
            'drug': self.drug_proj(x['drug']),
            'disease': self.disease_proj(x['disease']),
        }

        drug_embs = [h['drug']]
        disease_embs = [h['disease']]

        # Layer 1
        h = self.gat1(
            g,
            h,
            apply_norm=True,
            apply_dropout=True
        )

        drug_embs.append(h['drug'])
        disease_embs.append(h['disease'])

        # Layer 2
        h = self.gat2(
            g,
            h,
            apply_norm=True,
            apply_dropout=True
        )

        drug_embs.append(h['drug'])
        disease_embs.append(h['disease'])

        # =========================================================================
        # LLM branch
        # =========================================================================

        drug_llm_embs = []
        disease_llm_embs = []

        if self.model_type in ['graph_graph', 'cross_graph']:

            h_llm = {
                'drug': self.drug_proj_llm(x['drug_LLM']),
                'disease': self.disease_proj_llm(x['disease_LLM']),
            }

            drug_llm_embs.append(h_llm['drug'])
            disease_llm_embs.append(h_llm['disease'])

            h_llm = self.gat_llm1(
                g_llm,
                h_llm,
                apply_norm=True,
                apply_dropout=True
            )

            drug_llm_embs.append(h_llm['drug'])
            disease_llm_embs.append(h_llm['disease'])

            h_llm = self.gat_llm2(
                g_llm,
                h_llm,
                apply_norm=True,
                apply_dropout=True
            )

            drug_llm_embs.append(h_llm['drug'])
            disease_llm_embs.append(h_llm['disease'])

        elif self.model_type == 'graph_ae':

            h_llm = {
                'drug': self.drug_proj_llm(x['drug_LLM']),
                'disease': self.disease_proj_llm(x['disease_LLM']),
            }

            drug_llm_embs.append(h_llm['drug'])
            disease_llm_embs.append(h_llm['disease'])

            for key in ['drug', 'disease']:

                h_llm[key] = self.mlp_activation(
                    self.mlp_llm1(h_llm[key])
                )

            drug_llm_embs.append(h_llm['drug'])
            disease_llm_embs.append(h_llm['disease'])

            for key in ['drug', 'disease']:

                h_llm[key] = self.mlp_activation(
                    self.mlp_llm2(h_llm[key])
                )

            drug_llm_embs.append(h_llm['drug'])
            disease_llm_embs.append(h_llm['disease'])

        # =========================================================================
        # Layer attention pooling
        # =========================================================================

        drug_stack = torch.stack(drug_embs, dim=1)
        disease_stack = torch.stack(disease_embs, dim=1)

        h_final = {

            'drug': self.layer_att_drug(drug_stack),

            'disease': self.layer_att_disease(disease_stack),
        }

        # =========================================================================
        # Merge LLM branch
        # =========================================================================

        if len(drug_llm_embs) > 0:

            llm_drug_stack = torch.stack(drug_llm_embs, dim=1)
            llm_disease_stack = torch.stack(disease_llm_embs, dim=1)

            h_final['drug'] = (
                h_final['drug']
                + 0.5 * self.layer_att_drug_llm(llm_drug_stack)
            )

            h_final['disease'] = (
                h_final['disease']
                + 0.5 * self.layer_att_disease_llm(llm_disease_stack)
            )

        # =========================================================================
        # Decode
        # =========================================================================

        return self.decoder(h_final)