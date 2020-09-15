import torch
import os
import sys
import argparse
from argparse import Namespace
import numpy as np
import random


parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='train', help='train | test')
parser.add_argument('--data_dir', type=str, default='dataset/')
parser.add_argument('--glove_file', type=str, default='glove.6B.300d.txt')
parser.add_argument('--ckpt_path', type=str, default='')
parser.add_argument('--comment', type=str, default='')

parser.add_argument('--embed_dim', type=int, default=300, help='Word embedding dimension.')
parser.add_argument('--ner_embed_dim', type=int, default=30, help='NER embedding dimension. concat with word embedding on dim2')
parser.add_argument('--pos_embed_dim', type=int, default=30, help='POS embedding dimension. concat with word embedding on dim2')
parser.add_argument('--lgcn_hidden_dim', type=int, default=200, help='Local GCN hidden size.')
parser.add_argument('--input_dropout', type=float, default=0.2, help='Input dropout rate for word embeddings')
parser.add_argument('--tune_topk', type=int, default=1e10, help='Only finetune top N word embeddings.')
parser.add_argument('--lower', dest='lower', action='store_true', default=True, help='Lowercase all words.')

parser.add_argument('--pool_type', choices=['max', 'avg', 'sum'], default='max', help='Pooling function type for local gcn. Default max.')

parser.add_argument('--rnn_hidden_dim', type=int, default=200, help='RNN hidden state size.')
parser.add_argument('--rnn_layers', type=int, default=2, help='Num of RNN layers.')
parser.add_argument('--rnn_dropout', type=float, default=0.2, help='RNN dropout rate.')

parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--max_lr', type=float, default=1e-3, help='maximum learning rate for cyclic learning rate')
parser.add_argument('--base_lr', type=float, default=5e-5, help='minimum learning rate for cyclic learning rate')
parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam', 'adamw', 'adamax'], default='adam', help='Optimizer: sgd, adamw, adamax, adam')
parser.add_argument('--scheduler', type=str, choices=['', 'exp', 'cyclic'], default='', help='use scheduler')
parser.add_argument('--lr_decay', type=float, default=0.98, help='scheduler decay')
parser.add_argument('--num_epoch', type=int, default=100, help='Number of total training epochs.')
parser.add_argument('--batch_size', type=int, default=16, help='batch size cuda can support')
parser.add_argument('--actual_batch_size', type=int, default=16, help='actual batch size that you want')
parser.add_argument('--save_dir', type=str, default='lightning_logs', help='Root dir for saving models.')

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--model_file', type=str, help='Filename of the pretrained model.')

parser.add_argument('--glstm_hidden_dim', type=int, default=128, help='size of global lstm hidden state [default: 128]')
parser.add_argument('--glstm_layers', type=int, default=2, help='Number of global lstm layers [default: 2]')
parser.add_argument('--glstm_dropout_prob', type=float, default=0.1,help='recurrent dropout prob [default: 0.1]')

parser.add_argument('--ggcn_n_iter', type=int, default=1, help='iteration hop [default: 1] for global GCN')
parser.add_argument('--ggcn_layers', type=int, default=1, help='Number of GAT layers [default: 1]')
parser.add_argument('--ggcn_hidden_size', type=int, default=300, help='final output size & sentence node hidden size [default: 300]')
parser.add_argument('--edge_embed_size', type=int, default=50, help='feature embedding size for edge[default: 50]')
parser.add_argument('--ffn_inner_hidden_size', type=int, default=512,help='PositionwiseFeedForward inner hidden size [default: 512]')
parser.add_argument('--word2sent_n_head', type=int, default=10, help='multihead attention number [default: 10]')
parser.add_argument('--sent2word_n_head', type=int, default=10, help='multihead attention number [default: 10]')
parser.add_argument('--atten_dropout_prob', type=float, default=0.1, help='attention dropout prob [default: 0.1]')
parser.add_argument('--ffn_dropout_prob', type=float, default=0.1,help='PositionwiseFeedForward dropout prob [default: 0.1]')


parser.add_argument('--dropout_rate', type=float, default=0.3, help="dropout rate for classifier")

parser.add_argument('--rm_stopwords', type=bool, default=False, help='Remove stopwords in global word Node')

# # Ablation arguments
# parser.add_argument('--use_ner_node', type=bool, default=True, help='Use NER node in global model')
# 
# parser.add_argument('--riu', default=False, action='store_true', help='Randomly initialize Utterance Node')
# parser.add_argument('--riw', default=False, action='store_true', help='Randomly initialize Word Node')
# parser.add_argument('--riwwte', default=False, action='store_true', help='Randomly initialize Word Node With Trainable Embedding')
# 
# parser.add_argument('--use_rnn', type=bool, default=True, help='Use Local LSTM')
# parser.add_argument('--no_lgcn', type=bool, default=False, help="No local gcn")
# parser.add_argument('--local_pool_method', type=str, default='max', help='Pooling method for representation after local model') # this will only work if no_lgcn=True
# parser.add_argument('--no_glstm', type=bool, default=False, help='no global lstm')
# 
# 
# parser.add_argument('--utter_first', type=bool, default=True, help='Update UtterNode first before update wordNode')
# 
# parser.add_argument('--add_argNodes', type=bool, default=True, help='Add argNodes')
# parser.add_argument('--add_argEmb', type=bool, default=False, help='Add Argument embedding on gloabl model')
# parser.add_argument('--add_speakerNodes', type=bool, default=True, help='Add speaker nodes')
# 
# parser.add_argument('--pos_initEdge', type=bool, default=False, help='Use pos tag to initialize edge between word and sentence')
# parser.add_argument('--sep_update', type=bool, default=False, help='Separate update ner-word and utter-word')
args = parser.parse_args()



class Config:
    def __init__(self, args):
        self.__dict__.update(vars(args))
        self.gcn_lin_dim = self.lgcn_hidden_dim
        self.ws_edge_bucket = 10
        self.wn_edge_bucket = 10
        self.train_f, self.val_f, self.test_f = (os.path.join(self.data_dir, o) for o in ['train.json', 'dev.json', 'test.json'])
        self.glove_f = os.path.join(self.data_dir, self.glove_file)
        self.embed_f = os.path.join(self.data_dir, 'embeddings.npy')        
        self.proce_f = os.path.join(self.data_dir, 'dataset_preproc.p')
        self.proce_f_c = os.path.join(self.data_dir, 'dataset_preproc_c.p')
        self.num_workers = self.batch_size//2 
        self.gpus = 1
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_nodes = 1
        self.precision = 32
        if self.mode == 'test': assert len(self.ckpt_path) > 0, "Please provide a --ckpt_path"

config = Config(args)
## # >>>>>>>>>>> MODEL >>>>>>>>>>>>>>>>>
## # Embedding
## embed_dim = args.embed_dim
## tune_topk = args.topk
## ner_embed_dim = args.ner_embeded_dim
## pos_embed_dim = args.pos_embeded_dim
## input_dropout = args.input_dropout 
## use_pos_emb = args.use_pos_emb
## use_ner_emb = args.use_ner_emb
## 
## # RNN
## rnn_hidden_dim = args.rnn_hidden_dim
## rnn_layers = args.rnn_layers
## rnn_dropout = args.rnn_dropout
## 
## # localGCN
## gcn_dropout = args.gcn_dropout
## lgcn_hidden_dim = args.lgcn_hidden_dim
## lgcn_num_layers = args.lgcn_num_layers
## no_adj = args.no_adj
## prune_k = args.prune_k # qp set as 1
## conv_l2 = args.conv_l2
## 
## # MLP layers
## gcn_lin_dim = lgcn_hidden_dim
## mlp_layers = args.mlp_layers # qp
## pool_type = args.pool_type
## pooling_l2 = args.pooling_l2 # qp set as 0.003
## 
## # Global LSTM
## glstm_hidden_dim = args.glstm_hidden_dim
## glstm_layers = args.glstm_layers
## glstm_dropout_prob = args.glstm_dropout_prob
## 
## # Global GCN
## ws_edge_bucket = 10
## wn_edge_bucket = 10
## ggcn_n_iter = args.ggcn_n_iter
## ggcn_layers = args.ggcn_layers
## ggcn_hidden_size = args.ggcn_hidden_size
## edge_embed_size = args.edge_embed_size
## ffn_inner_hidden_size = args.ffn_inner_hidden_size 
## sent2word_n_head = args.sent2word_n_head
## word2sent_n_head = args.word2sent_n_head
## atten_dropout_prob = args.atten_dropout_prob 
## ffn_dropout_prob = args.ffn_dropout_prob
## use_ner_node = args.use_ner_node
## 
## # Final Classifier
## dropout_rate = args.dropout_rate
## 
## # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
## 
## 
## # >>>>>>>>>> OTHERS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
## # data
## train_f, val_f, test_f = [os.path.join(args.data_dir, o) for o in ['train.json', 'dev.json', 'test.json']]
## glove_f = os.path.join(args.data_dir, args.glove_file)
## embed_f = 'cache/embeddings.npy'
## proce_f = 'cache/dataset_preproc.p'
## proce_f_c = 'cache/dataset_preproc_c.p'
## 
## # preprocessing
## lower = args.lower
## min_freq = 0
## max_seq_len = 100
## max_num_utt = 50
## word_dropout = args.word_dropout # rate at which randomly select a word to UNK # 0.04 # qp
## 
## # training
## batch_size = args.batch_size
## actual_batch_size = args.actual_batch_size
## lr = args.lr
## max_lr = args.max_lr
## base_lr = args.base_lr
## num_epoch = args.num_epoch
## optimizer = args.optimizer
## scheduler = args.scheduler
## lr_decay = args.lr_decay
## 
## # save 
## save_dir = args.save_dir
## 
## # resources
## num_workers = batch_size//2 
## gpus = 1 # TODO: only allow 1 gpu for now
## device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # TODO: use gpus instead
## num_nodes = 1
## precision = 32 # TODO: half precision currently not supported with dgl
## 
## 
## # others
## seed = args.seed
## mode = args.mode
## ckpt_path = args.ckpt_path
## comment = args.comment
## if mode == 'test': assert len(ckpt_path) > 0, "Please provide a --ckpt_path"
## 
## # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
## 
## 
## # >>>>>> Ablation >>>>>>>>>>>>>>>>>>>>>>>>>>>>
## riu = args.riu
## riw = args.riw
## riwwte = args.riwwte
## 
## use_rnn = args.use_rnn
## no_lgcn = args.no_lgcn
## local_pool_method = args.local_pool_method
## no_glstm = args.no_glstm
## 
## utter_first = args.utter_first
## rm_stopwords = args.rm_stopwords
## add_argNodes = args.add_argNodes
## add_argEmb = args.add_argEmb
## add_speakerNodes = args.add_speakerNodes
## 
## pos_initEdge = args.pos_initEdge
## sep_update = args.sep_update
## # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
