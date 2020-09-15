import torch
import torch.utils.data as data
import random, os, re, ast, time
import numpy as np
import collections
import dgl
from dgl.data.utils import save_graphs, load_graphs
from nltk.corpus import stopwords

from utils.torch_utils import get_positions
from utils.config import config
from utils import constant

stopwords = stopwords.words('english')


class Dataset(data.Dataset):
    def __init__(self, data, vocab):
        self.vocab = vocab
        self.data = data 


    def __len__(self):
        return len(self.data)


    def conv_to_ids(self, conv):
        conv_ids = [torch.LongTensor(self.vocab.map(o)) for o in conv]
        return conv_ids


    def label_to_oneHot(self, rel_labels):
        rid = []
        # does not consider relation id 37 ("unanswerable")
        for k in range(1,37):
            rid += [1] if k in rel_labels else [0]
        return torch.FloatTensor(rid)


    def remove_stopwords(self, utter): return [w for w in utter if w not in stopwords]


    def add_wordNode(self, G, num_nodes, word_ids, x_wids=None, y_wids=None):
        """word: unit=0, dtype=0
        """
        G.add_nodes(num_nodes)
        wid2nid = {w:i for i,w in enumerate(word_ids)}
        nid2wid = {i:w for w,i in wid2nid.items()}
        G.ndata['unit'] = torch.zeros(num_nodes)
        G.ndata['dtype'] = torch.zeros(num_nodes)
        wordids = list(sorted(wid2nid.keys(), key=lambda o: wid2nid[o]))
        G.ndata['id'] = torch.LongTensor(wordids) # id means wordid
        return wid2nid, nid2wid

    def add_sentNode(self, G, num_nodes, start_ids):
        """sent: unit=1, dtype=1"""
        G.add_nodes(num_nodes)
        sid2nid = {i: i+start_ids for i in range(num_nodes)}
        nid2sid = {i+start_ids: i for i in range(num_nodes)}
        G.ndata['unit'][start_ids:] = torch.ones(num_nodes)
        G.ndata['dtype'][start_ids:] = torch.ones(num_nodes)
        return sid2nid, nid2sid

    def add_speakerNode(self, G, num_nodes, start_ids):
        """speaker: unit=0, dtype=3"""
        G.add_nodes(num_nodes)
        speakerid2nid = {i: i+start_ids for i in range(num_nodes)}
        nid2speakerid = {i+start_ids: i for i in range(num_nodes)}
        G.ndata['unit'][start_ids:] = torch.zeros(num_nodes)
        G.ndata['dtype'][start_ids:] = torch.ones(num_nodes) * 3
        G.ndata['id'][start_ids:] = torch.arange(num_nodes)
        return speakerid2nid, nid2speakerid


    def add_nerNode(self, G, num_nodes, start_ids, ner_ids):
        """ner: unit=1, dtype=2"""
        G.add_nodes(num_nodes)
        nerid2nid = {ner_id: i+start_ids for i, ner_id in enumerate(ner_ids)}
        nid2nerid = {i:n for n,i in nerid2nid.items()}
        G.ndata['unit'][start_ids:] = torch.ones(num_nodes)
        G.ndata['dtype'][start_ids:] = torch.ones(num_nodes) * 2
        G.ndata['id'][start_ids:] = torch.LongTensor( list(sorted(nerid2nid.keys(), key=lambda o: nerid2nid[o]))) # id means nerid
        return nerid2nid, nid2nerid

    def add_argNode(self, G, num_nodes, start_ids):
        """arg: unit=0, dtype=4"""
        G.add_nodes(num_nodes)
        argid2nid = {i: i+start_ids for i in range(num_nodes)}
        nid2argid = {v:k for k,v in argid2nid.items()}
        G.ndata['unit'][start_ids:] = torch.zeros(num_nodes)
        G.ndata['dtype'][start_ids:] = torch.ones(num_nodes) * 4
        G.ndata['id'][start_ids:] = torch.arange(num_nodes)
        return argid2nid, nid2argid

    def get_weight(self): 
        return torch.randint(config.ws_edge_bucket, (1,) )

    def creat_graph(self, utters, utters_ner, x, y, pos_tags):
        """create graph for each conversation
        Parameters:
        -----------
        utters: list[list[str]]
        utters_ner: list[list[str]]
            list of utterance's ner

        Returns:
        --------
        graph: dgl.DGLGraph
            node:
                word : unit=0, dtype=0
                sent: unit=1, dtype=1
                ner: unit=1, dtype=2
                speaker: unit=0, dtype=3 
                arg: unit=0, dtype=4
            edge:
                word & speaker & arg - sent: dtype=0
                word & arg - ner: dtype=1
        """
        G = dgl.graph([])
        G.set_n_initializer(dgl.init.zero_initializer)
        
        # word - sent
        utters_ids = self.conv_to_ids(utters)
        if config.rm_stopwords:
            words = [w for u in utters for w in self.remove_stopwords(u)]
        else:
            words = [w for u in utters for w in u]
        words_ids = sorted(list(set(self.vocab.map(words))))
        wid2nid, nid2wid = self.add_wordNode(G, len(words_ids), words_ids)
        sid2nid, nid2sid = self.add_sentNode(G, len(utters_ids), len(words_ids))
        
        # Init edge with POS
        wordUtter_pos_dict = {}
        for u_i, (w_ids, pos_ids) in enumerate(zip(utters_ids, pos_tags)):
            for w_i, pos_i in zip(w_ids, pos_ids): wordUtter_pos_dict[(u_i, int(w_i))] = torch.tensor([pos_i])

        for sid in range(len(utters_ids)):
            for wid in words_ids:
                u_ids = utters_ids[sid]
                if wid in u_ids:
                    ws_link = wordUtter_pos_dict[(sid, wid)]
                    G.add_edges(wid2nid[wid], sid2nid[sid], data={'ws_link': ws_link, "dtype": torch.tensor([0])} )
                    G.add_edges(sid2nid[sid], wid2nid[wid], data={'ws_link': ws_link, "dtype": torch.tensor([0])} )

        # get the node of x and y
        x_node_id = [wid2nid[xids] for xids in self.vocab.map(x.lower().split()) if xids in wid2nid]
        y_node_id = [wid2nid[yids] for yids in self.vocab.map(y.lower().split()) if yids in wid2nid]
        if len(x_node_id) == 0: x_node_id = [np.random.choice( list(wid2nid.values()) )]
        if len(y_node_id) == 0: y_node_id = [np.random.choice( list(wid2nid.values()) )]
        
        # word - NER type
        utters_ner = [[constant.ner_s2i[oo] for oo in o] for o in utters_ner]
        word_ner_dict = collections.defaultdict(list)
        for uid, u_nerid in zip(utters_ids, utters_ner):
            for wid, nerid in zip(uid, u_nerid):
                word_ner_dict[wid].append(nerid)
        ner_ids = sorted(list(set([oo for o in utters_ner for oo in o])))
        nerid2nid, nid2nerid = self.add_nerNode(G, len(ner_ids), len(words_ids) + len(utters_ids), ner_ids )
        for wid in words_ids:
            word_ner_ids = word_ner_dict[wid]
            for nerid in word_ner_ids:
                G.add_edges(wid2nid[wid], nerid2nid[nerid], data={'wn_link': self.get_weight(), "dtype": torch.tensor([1])} )
                G.add_edges(nerid2nid[nerid], wid2nid[wid],  data={'wn_link': self.get_weight(), "dtype": torch.tensor([1])} )

        # Speaker Nodes
        # speaker - sent
        speaker_ids = [int(o[0][-1])-1 if o[0][-1].isdigit() else int(o[0][-2])-1 for o in utters] # speaker id start from 0
        num_speaker_node = max(speaker_ids) + 1
        speakerid2nid, nid2speakerid = self.add_speakerNode(G, num_speaker_node, len(words_ids) + len(utters_ids) + len(ner_ids))
        # here sid stands for sentenceid and speakerid stands for speaker id
        for sid, speakerid in enumerate(speaker_ids):
            G.add_edges(sid2nid[sid], speakerid2nid[speakerid], data={'ws_link': self.get_weight(), "dtype": torch.tensor([0])})
            G.add_edges(speakerid2nid[speakerid], sid2nid[sid], data={'ws_link': self.get_weight(), "dtype": torch.tensor([0])})

        # Entity-type nodes
        # argument - sent edge
        argid2nid, nid2argid = self.add_argNode(G, 2, len(words_ids)+len(utters_ids)+len(ner_ids)+num_speaker_node)

        arga_wordid = [nid2wid[o] for o in x_node_id]
        arga_sentid = []
        for wid in arga_wordid:
            for sid in range(len(utters_ids)):
                if wid in utters_ids[sid]: arga_sentid.append(sid)
        for sid in sorted(list(set(arga_sentid))):
            G.add_edges(sid2nid[sid], argid2nid[0], data={'ws_link': self.get_weight(), "dtype": torch.tensor([0])} )
            G.add_edges(argid2nid[0], sid2nid[sid], data={'ws_link': self.get_weight(), "dtype": torch.tensor([0])} )

        argb_wordid = [nid2wid[o] for o in y_node_id]
        argb_sentid = []
        for wid in argb_wordid:
            for sid in range(len(utters_ids)):
                if wid in utters_ids[sid]: argb_sentid.append(sid)
        for sid in sorted(list(set(argb_sentid))):
            G.add_edges(sid2nid[sid], argid2nid[1], data={'ws_link': self.get_weight(), "dtype": torch.tensor([0])} )
            G.add_edges(argid2nid[1], sid2nid[sid], data={'ws_link': self.get_weight(), "dtype": torch.tensor([0])} )

        # argument - ner edge
        # argument - ner
        for wid in arga_wordid:
            for nerid in word_ner_dict[wid]:
                G.add_edges(nerid2nid[nerid], argid2nid[0], data={'ws_link': self.get_weight(), "dtype": torch.tensor([1])})
                G.add_edges(argid2nid[0], nerid2nid[nerid], data={'ws_link': self.get_weight(), "dtype": torch.tensor([1])})
        
        for wid in argb_wordid:
            for nerid in word_ner_dict[wid]:
                G.add_edges(nerid2nid[nerid], argid2nid[1], data={'ws_link': self.get_weight(), "dtype": torch.tensor([1])})
                G.add_edges(argid2nid[1], nerid2nid[nerid], data={'ws_link': self.get_weight(), "dtype": torch.tensor([1])})



        return x_node_id, y_node_id, G

    def __getitem__(self, index):
        """
        .. note:: `utter` and `u` both stands for utterance
        """
        item = {}

        item["utters"] = self.conv_to_ids(self.data[index]['feats']['tokens'])
        item["u_lengths"] = [len(o) for o in item["utters"]]
        item["u_masks"] = [torch.LongTensor([1 for _ in range(o)]) for o in item["u_lengths"]]
        item['rids'] = self.label_to_oneHot(self.data[index]['rid'])
        item['ner_type'] = [torch.LongTensor([constant.ner_s2i[oo] for oo  in o]) for o in self.data[index]['feats']['ner_type'] ]
        item['pos_tag'] = [torch.LongTensor([constant.pos_s2i[oo] for oo in o])  for o in self.data[index]['feats']['pos_tag'] ]
        item['x_node_id'], item['y_node_id'], item['conv_graph'] = self.creat_graph(
            self.data[index]['feats']['tokens'], 
            self.data[index]['feats']['ner_type'],
            self.data[index]['x'],
            self.data[index]['y'], 
            [ [constant.pos_s2i[oo] for oo in o] for o in self.data[index]['feats']['pos_tag'] ],
        ) 
        return item

    def get_arg_pos(self, arg, conv):
        arg_positions = []
        arg = arg.lower().split()
        for utter in conv:
            for i, w in enumerate(utter):
                if w == arg[0]:
                    arg_positions.append( torch.LongTensor(get_positions(i, i+len(arg)-1, len(utter))) )
                    break
                elif i == len(utter) - 1:
                    arg_positions.append( torch.LongTensor(get_positions(1, 0, len(utter))) )

        return arg_positions



def collate_fn(data):
    """
    .. note:: `utter` for utterance, `conv` for conversation, `seq` for sequence
    """
    items = {}
    for k in data[0].keys():
        items[k] = [d[k] for d in data]
    
    num_utters = [len(conv) for conv in items['utters']] 
    max_utters = max(num_utters)
    max_seq = max([utter_l for conv_l in items['u_lengths'] for utter_l in conv_l])

    def pad(convs, conv_lengths=items['u_lengths'], max_utters=max_utters, max_seq=max_seq):
        """
        Parameters
        ----------
        convs: list[list[list[int]]]
        conv_lengths: list[list[int]] 
        max_utters: int
            the max number of utterance in one conversation
        max_seq: int
            the max number of words in one utterance
        
        Returns
        -------
        padded_convs: (batch size, max number of utterance, max sequence length)
        .. note:: pad index is 0
        """
        padded_convs = torch.zeros(len(convs), max_utters, max_seq, dtype=torch.int64)
        for b_i, (conv_l, conv) in enumerate(zip(conv_lengths, convs)):
            for u_i, (utter_l, utter) in enumerate(zip(conv_l, conv)):
                padded_convs[b_i, u_i, :utter_l] = utter
        return padded_convs #.to(config.device)

    def pad_length(conv_lengths):
        """
        conv_lengths: list[list[int]] 
        """
        padded_lengths = torch.zeros( (len(conv_lengths), max_utters), dtype=torch.int64)
        for bi, conv_l in enumerate(conv_lengths):
            for ui, u_l in enumerate(conv_l):
                padded_lengths[bi, ui] = u_l
        return padded_lengths

    d = {}
    d['utter_index'] = [oo + i*max_utters for i, o in enumerate(num_utters) for oo in range(o)]
    d['conv_batch'], d['conv_mask'] = pad(items['utters']), pad(items['u_masks'])
    d['ner_type'], d['pos_tag'] = pad(items['ner_type']), pad(items['pos_tag'])
    d['conv_lengths'] = pad_length(items['u_lengths'])
    d['rids'] = torch.stack(items['rids'])
    d['batch_graph'] = dgl.batch(items['conv_graph'])
    d['x_node_id'], d['y_node_id'] = items['x_node_id'], items['y_node_id']

    return d


