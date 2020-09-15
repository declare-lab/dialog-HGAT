# Some of the classes are adapted from brxx12/HeterSumGraph

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class WSWGAT(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, attn_drop_out, ffn_inner_hidden_size, ffn_drop_out, feat_embed_size, layerType):
        super().__init__()
        self.layerType = layerType
        if layerType == "W2S":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, layer=WSGATLayer)
        elif layerType == "S2W":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, layer=SWGATLayer)
        elif layerType == "S2S":
            self.layer = MultiHeadSGATLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out)
        else:
            raise NotImplementedError("GAT Layer has not been implemented!")

        self.ffn = PositionwiseFeedForward(out_dim, ffn_inner_hidden_size, ffn_drop_out)

    def forward(self, g, w, s):
        if self.layerType == "W2S":
            origin, neighbor = s, w
        elif self.layerType == "S2W":
            origin, neighbor = w, s
        elif self.layerType == "S2S":
            assert torch.equal(w, s)
            origin, neighbor = w, s
        else:
            origin, neighbor = None, None

        h = F.elu(self.layer(g, neighbor))
        h = h + origin
        h = self.ffn(h.unsqueeze(0)).squeeze(0)
        return h


class MultiHeadSGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, attn_drop_out, merge='cat'):
        super(MultiHeadSGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(
                SGATLayer(in_dim, out_dim))  # [n_nodes, hidden_size]
        self.merge = merge
        self.dropout = nn.Dropout(attn_drop_out)

    def forward(self, g, h):
        head_outs = [attn_head(g, self.dropout(h)) for attn_head in self.heads]  # n_head * [n_nodes, hidden_size]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)  # [n_nodes, hidden_size * n_head]
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))


class MultiHeadLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, attn_drop_out, feat_embed_size, layer, merge='cat'):
        super(MultiHeadLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(layer(in_dim, out_dim, feat_embed_size))  # [n_nodes, hidden_size]
        self.merge = merge
        self.dropout = nn.Dropout(attn_drop_out)

    def forward(self, g, h):
        head_outs = [attn_head(g, self.dropout(h)) for attn_head in self.heads]  # n_head * [n_nodes, hidden_size]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            result = torch.cat(head_outs, dim=1)  # [n_nodes, hidden_size * n_head]
        else:
            # merge using average
            result = torch.mean(torch.stack(head_outs))
        return result


class SGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, weight=0):
        super(SGATLayer, self).__init__()
        self.weight = weight
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)  # [edge_num, 2 * out_dim]
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h):
        snode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        sedge_id = g.filter_edges(lambda edges: edges.data["dtype"] == 0)
        z = self.fc(h)
        g.nodes[snode_id].data['z'] = z
        g.apply_edges(self.edge_attention, edges=sedge_id)
        g.pull(snode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        h = g.ndata.pop('sh')
        return h[snode_id]


class WSGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_embed_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)

        self.feat_fc = nn.Linear(feat_embed_size, out_dim, bias=False)
        self.attn_fc = nn.Linear(3 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        dfeat = self.feat_fc(edges.data["ws_embed"])                  # [edge_num, out_dim]
        z2 = torch.cat([edges.src['z'], edges.dst['z'], dfeat], dim=1)  # [edge_num, 3 * out_dim]
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}

    def message_func(self, edges):
        # print("edge e ", edges.data['e'].size())
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h):
        wnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
        snode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        wsedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 0) & (edges.dst["unit"] == 1))
        # print("id in WSGATLayer")
        # print(wnode_id, snode_id, wsedge_id)
        z = self.fc(h)
        g.nodes[wnode_id].data['z'] = z
        g.apply_edges(self.edge_attention, edges=wsedge_id)
        g.pull(snode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        h = g.ndata.pop('sh')
        return h[snode_id]


class SWGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_embed_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)

        self.feat_fc = nn.Linear(feat_embed_size, out_dim)
        self.attn_fc = nn.Linear(3 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        dfeat = self.feat_fc(edges.data["ws_embed"])  # [edge_num, out_dim]
        z2 = torch.cat([edges.src['z'], edges.dst['z'], dfeat], dim=1)  # [edge_num, 3 * out_dim]
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h):
        wnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
        snode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        swedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 1) & (edges.dst["unit"] == 0))
        z = self.fc(h)
        g.nodes[snode_id].data['z'] = z
        g.apply_edges(self.edge_attention, edges=swedge_id)
        g.pull(wnode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        h = g.ndata.pop('sh')
        return h[wnode_id]


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        assert not torch.any(torch.isnan(x)), "FFN input"
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        assert not torch.any(torch.isnan(output)), "FFN output"
        return output




