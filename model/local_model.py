import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import constant
from utils import torch_utils





class LocalLSTM(nn.Module):
    """
    A GCN/Contextualized GCN module operated on dependency graphs.
    modified module taken from qipeng/gcn-over-pruned-trees
    """
    def __init__(self, config, embeddings):
        super().__init__()
        self.config = config
        self.in_dim = config.embed_dim + config.pos_embed_dim + config.ner_embed_dim
        self.emb, self.pos_emb, self.ner_emb = embeddings
        self.in_drop = nn.Dropout(config.input_dropout)

        # rnn layer
        input_size = self.in_dim
        self.rnn = nn.LSTM(input_size, config.rnn_hidden_dim, config.rnn_layers, batch_first=True, dropout=config.rnn_dropout, bidirectional=True)
        self.in_dim = config.rnn_hidden_dim * 2
        self.rnn_drop = nn.Dropout(config.rnn_dropout) # use on last layer output

        self.linear = nn.Linear(self.in_dim, config.lgcn_hidden_dim)

    def conv_l2(self):
        conv_weights = []
        for w in self.W:
            conv_weights += [w.weight, w.bias]
        return sum([x.pow(2).sum() for x in conv_weights])

    def encode_with_rnn(self, rnn_inputs, masks, h_states=None):
        rnn_inputs = rnn_inputs * masks.unsqueeze(-1)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, h_states) 
        rnn_outputs = rnn_outputs * masks.unsqueeze(-1) # (batch * num_utt, num_words, rnn_hid * 2)
        return rnn_outputs

    def forward(self, input_batch):
        """
        :words: (batch*num_utter, num_words)
        :masks: (batch*num_utter, num_words)
        :pos: (batch*num_utter, num_words)
        :ner: (batch*num_utter, num_words)
        """
        words, masks, pos, ner = input_batch
        word_embs = self.emb(words)
        embs = [word_embs]

        if self.config.pos_embed_dim > 0:
            embs += [self.pos_emb(pos)]
        if self.config.ner_embed_dim > 0:
            embs += [self.ner_emb(ner)]
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)

        # rnn layer
        gcn_inputs = self.rnn_drop(self.encode_with_rnn(embs, masks))
        
        return self.linear(gcn_inputs) * masks.unsqueeze(-1)
    




class UtterEncoder(nn.Module):
    """
    Encoder for each utterance, Wrapper class for localGCN
    """
    def __init__(self, config, vocab):
        super().__init__()
        self.config = config
        self.vocab = vocab
        # Embedding
        self.embedding = nn.Embedding(self.vocab.n_words, self.config.embed_dim)
        self.pos_embedding = nn.Embedding(len(constant.pos_i2s), self.config.pos_embed_dim) if self.config.pos_embed_dim > 0 else None
        self.ner_embedding = nn.Embedding(len(constant.ner_i2s), self.config.ner_embed_dim) if self.config.ner_embed_dim > 0 else None
        self.init_pretrained_embeddings_from_numpy(np.load(open(config.embed_f, 'rb'), allow_pickle=True))
        self.embeddings = (self.embedding, self.pos_embedding, self.ner_embedding)
        self.lstm = LocalLSTM(config, self.embeddings)

    def init_pretrained_embeddings_from_numpy(self, pretrained_word_vectors):
        self.embedding.weight = nn.Parameter(torch.from_numpy(pretrained_word_vectors).float())
        if self.config.tune_topk <= 0:
            print("Do not fine tune word embedding layer")
            self.embedding.weight.requires_grad = False
        elif self.config.tune_topk < self.vocab.n_words:
            print(f"Finetune top {self.config.tune_topk} word embeddings")
            self.embedding.weight.register_hook(lambda x: torch_utils.keep_partial_grad(x, self.config.tune_topk))
        else: print("Finetune all word embeddings")


    def forward(self, batch):
        """
        :conv_batch: (batch, num_utt, num_words)
        :conv_mask: (batch, num_utt, num_words)
        """
        conv_batch, conv_mask, conv_lengths, ner_batch, pos_batch = batch['conv_batch'], batch['conv_mask'], batch['conv_lengths'], batch['ner_type'], batch['pos_tag']
        batch, num_utt, num_words = conv_batch.shape
        conv_batch, ner_batch, pos_batch, conv_mask = conv_batch.view(-1,num_words), ner_batch.view(-1,num_words), pos_batch.view(-1,num_words), conv_mask.view(-1,num_words) # (num_utt*batch, num_words)
        
        h = self.lstm((conv_batch, conv_mask, pos_batch, ner_batch))
        h_p = torch.max(h, dim=1)[0]

        utter_rep = h_p.reshape(batch, num_utt, -1)
        return utter_rep


