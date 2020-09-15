import numpy as np
import torch
import torch
import torch.nn as nn
import dgl

from model.global_model import ConvEncoder
from model.local_model import UtterEncoder


class Graph_DialogRe(nn.Module):

    def __init__(self, config, vocab):
        super().__init__()
        self.config = config
        self.vocab = vocab
        self.local_model = UtterEncoder(config, vocab)
        self.global_model = ConvEncoder(config, self.local_model.embeddings)
        self.classifier_dropout = nn.Dropout(config.dropout_rate)
        self.activation = nn.ReLU() # nn.Tanh()
        self.classifier = nn.Linear(config.embed_dim * 4, 36) # total 36 relations

    def forward(self, batch):
        graph = batch['batch_graph']
        x_node_id, y_node_id = batch['x_node_id'], batch['y_node_id']
        local_feature = self.local_model(batch) # (batch, max_number_utt, gcn_lin_dim)
        self.global_model(graph, batch, local_feature)
        arga, argb = self.unbatch_graph(graph, x_node_id, y_node_id) # (batch, embed_dim)
        tempa, tempb = self.unbatch_arg(graph)
        arga = torch.cat([arga, tempa], dim=-1)
        argb = torch.cat([argb, tempb], dim=-1)
        y = torch.cat([arga, argb], dim=-1)
        y = self.classifier_dropout(y)
        y = self.classifier(y)
        return y

    def unbatch_graph(self, graph, x_node_id, y_node_id):
        g_list = dgl.unbatch(graph)
        arga, argb = [], []
        for x_id, y_id, g in zip(x_node_id, y_node_id, g_list):
            x_feat = g.nodes[x_id].data['feat'] # (number_of_x_word, embed_dim)
            y_feat = g.nodes[y_id].data['feat']
            x_feat, _ = torch.max(x_feat, dim=0) 
            y_feat, _ = torch.max(y_feat, dim=0)
        
            arga.append(x_feat)
            argb.append(y_feat)

        return torch.stack(arga), torch.stack(argb)

    def unbatch_arg(self, graph):
        g_list = dgl.unbatch(graph)
        arga, argb = [], []
        for g in g_list:
            arg_nodeid = g.filter_nodes(lambda nodes: nodes.data["dtype"] == 4)
            arg_feat = g.nodes[arg_nodeid].data['feat']
            x_feat, y_feat = arg_feat[0], arg_feat[1]
            arga.append(x_feat)
            argb.append(y_feat)

        return torch.stack(arga), torch.stack(argb)


