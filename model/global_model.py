import numpy as np
import torch
import torch
import torch.nn as nn
import dgl

from model.local_model import UtterEncoder
from model.gat import WSWGAT
from utils import constant

class ConvEncoder(nn.Module):
    """
    without sent2sent and add residual connection
    adapted from brxx122/hetersumgraph/higraph.py
    """
    def __init__(self, config, embeddings):
        super().__init__()

        self.config = config
        # self._embed = embed
        # self.embed_size = hps.word_emb_dim
        self.word_embedding, self.pos_embedding, _ = embeddings
        self.ner_embedding = nn.Embedding(len(constant.ner_i2s), self.config.ner_embed_dim)
        self.speaker_embedding = nn.Embedding(10, self.config.embed_dim) # assume there is only 10 speaker in the conversation
        self.arg_embedding = nn.Embedding(2, self.config.embed_dim)


        # sent node feature
        self.ws_embed = nn.Embedding(len(constant.pos_i2s), config.edge_embed_size) # bucket = 10
        self.wn_embed = nn.Embedding(config.wn_edge_bucket, config.edge_embed_size) # bucket = 10
        self.sent_feature_proj = nn.Linear(config.glstm_hidden_dim*2, config.ggcn_hidden_size, bias=False)
        self.ner_feature_proj = nn.Linear(config.ner_embed_dim, config.ggcn_hidden_size, bias=False)
        self.glstm = nn.LSTM(self.config.gcn_lin_dim, 
                                self.config.glstm_hidden_dim, 
                                num_layers=self.config.glstm_layers, dropout=0.1,
                                batch_first=True, bidirectional=True)


        # word -> sent
        self.word2sent = WSWGAT(in_dim=config.embed_dim,
                                out_dim=config.ggcn_hidden_size,
                                num_heads=config.word2sent_n_head,
                                attn_drop_out=config.atten_dropout_prob,
                                ffn_inner_hidden_size=config.ffn_inner_hidden_size,
                                ffn_drop_out=config.ffn_dropout_prob,
                                feat_embed_size=config.edge_embed_size, # for edge
                                layerType="W2S"
                                )

        # sent -> word
        self.sent2word = WSWGAT(in_dim=config.ggcn_hidden_size,
                                out_dim=config.embed_dim,
                                num_heads=config.sent2word_n_head, # make sure word_embedding divisible by this number 
                                attn_drop_out=config.atten_dropout_prob,
                                ffn_inner_hidden_size=config.ffn_inner_hidden_size,
                                ffn_drop_out=config.ffn_dropout_prob,
                                feat_embed_size=config.edge_embed_size, # for edge
                                layerType="S2W"
                                )

        # node classification
        self.wh = nn.Linear(config.ggcn_hidden_size, 2)

    def forward(self, graph, batch, local_feature):
        supernode_id = graph.filter_nodes(lambda nodes: nodes.data['unit'] == 1) # supernodes contains nerNode and sentNode

        # Initialize states
        self.set_wordNode_feature(graph)
        self.set_speakerNode_feature(graph)
        self.set_argNode_feature(graph)
        self.set_wordSentEdge_feature(graph)
        self.set_sentNode_feature(graph, batch, local_feature)    # [snode, glstm_hidden_dim] -> [snode, n_hidden_size]

        self.set_wordNerEdge_feature(graph)
        self.set_nerNode_feature(graph)

        wnode_id = graph.filter_nodes(lambda nodes: nodes.data["unit"]==0) # both word node and speaker node
        # the start state
        word_state = graph.nodes[wnode_id].data['embed']
        sent_state = graph.nodes[supernode_id].data['init_state']
        ner_state = graph.nodes[supernode_id].data['ner_init_state']
        word_state = self.sent2word(graph, word_state, sent_state)
        ner_state = self.word2sent(graph, word_state, ner_state)
        word_state = self.sent2word(graph, word_state, ner_state)
        for i in range(self.config.ggcn_n_iter):
            sent_state = self.word2sent(graph, word_state, sent_state)
            word_state = self.sent2word(graph, word_state, sent_state)
        graph.nodes[wnode_id].data["feat"] = word_state
        return None

    def set_wordNode_feature(self, graph):
        wnode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"]==0) # only word node
        wid = graph.nodes[wnode_id].data["id"]  # [n_wnodes]
        w_embed = self.word_embedding(wid)  # [n_wnodes, D]
        graph.nodes[wnode_id].data["embed"] = w_embed
        return w_embed

    def set_speakerNode_feature(self, graph):
        speakernode_id = graph.filter_nodes(lambda nodes: nodes.data['dtype']==3) # only speaker node
        speakerid = graph.nodes[speakernode_id].data['id']
        speaker_embed = self.speaker_embedding(speakerid)
        graph.nodes[speakernode_id].data["embed"] = speaker_embed

    def set_argNode_feature(self, graph):
        argnode_id = graph.filter_nodes(lambda nodes: nodes.data['dtype']==4) # only arg node
        argid = graph.nodes[argnode_id].data['id']
        arg_embed = self.arg_embedding(argid)
        graph.nodes[argnode_id].data['embed'] = arg_embed
    
    def set_wordSentEdge_feature(self, graph):
        # Intialize word sent edge
        wsedge_id = graph.filter_edges(lambda edges: edges.data["dtype"] == 0)   # both word node and speaker node
        ws_edge = graph.edges[wsedge_id].data['ws_link']
        graph.edges[wsedge_id].data["ws_embed"] = self.ws_embed(ws_edge)


    def set_sentNode_feature(self, graph, batch, local_feature):
        sent_feature, _ = self.glstm(local_feature) # (batch, max_number_utt, glstm_hidden_dim*2)
        sent_feature = sent_feature * batch['conv_mask'][:,:,0].unsqueeze(-1) # masking 
        sent_feature = sent_feature.reshape(-1, sent_feature.size(-1))
        sent_feature = sent_feature[batch['utter_index']] # (batch * total_number_utt, glstm_hidden_dim*2)
        
        snode_id = graph.filter_nodes(lambda nodes: nodes.data['dtype'] == 1) # only sent node
        
        graph.nodes[snode_id].data['init_state'] = self.sent_feature_proj(sent_feature)

    def set_nerNode_feature(self, graph):
        nnode_id = graph.filter_nodes(lambda nodes: nodes.data['dtype'] == 2) # only ner node
        nerid = graph.nodes[nnode_id].data['id'] # [n_nerNodes]
        ner_embed = self.ner_embedding(nerid)
        graph.nodes[nnode_id].data['ner_init_state'] = self.ner_feature_proj(ner_embed)

    def set_wordNerEdge_feature(self, graph):
        wnedge_id = graph.filter_edges(lambda edges: edges.data["dtype"] == 1) # only word node and NER node
        wn_edge = graph.edges[wnedge_id].data['ws_link']
        graph.edges[wnedge_id].data['wn_embed'] = self.wn_embed(wn_edge)

