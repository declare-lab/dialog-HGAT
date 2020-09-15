import sklearn
import os
from argparse import Namespace
import pickle
import torch
from torch import nn
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer


from utils.torch_utils import f1_score, acc_score, f1c_score
from model.model import Graph_DialogRe
from utils.data_reader import load_dataset, load_dataset_c, get_original_data
from utils.data_loader import collate_fn, Dataset


class Train_GraphDialogRe(LightningModule):
    """
    Pytorch-lightning wrapper class for training
    """
   
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        trn_data, val_data, tst_data, self.vocab = load_dataset()
        self.ds_trn, self.ds_val, self.ds_tst = Dataset(trn_data, self.vocab), Dataset(val_data, self.vocab), Dataset(tst_data, self.vocab)
        val_data_c, test_data_c = load_dataset_c()
        self.ds_val_c, self.ds_tst_c = Dataset(val_data_c, self.vocab), Dataset(test_data_c, self.vocab)
        exclude = ['np','torch','random','args', 'os', 'argparse', 'parser', 'Namespace', 'sys']
        self.hparams = Namespace(**{k:v for k,v in config.__dict__.items() if k[:2]!='__' and k not in exclude})
        self.model = Graph_DialogRe(config, self.vocab)
        self.loss_fn = nn.BCEWithLogitsLoss() # nn.BCELoss()
        self.f1_metric = f1_score
        self.f1c_metric = f1c_score
        self.acc_metric = acc_score

    
    def configure_optimizers(self):
        if self.config.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        elif self.config.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.config.lr)
        elif self.config.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr)
        elif self.config.optimizer == 'adamax':
            optimizer = torch.optim.Adamax(self.parameters(), lr=self.config.lr)
        if len(self.config.scheduler) == 0:
            return optimizer
        elif self.config.scheduler == 'exp':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.config.lr_decay)
            return [optimizer], [scheduler]
        elif self.config.scheduler == 'cyclic':
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, self.config.base_lr, self.config.max_lr, cycle_momentum=False)
            return [optimizer], [scheduler]

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        pred = self(batch)
        loss = self.loss_fn(pred, batch['rids'])
        tb_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tb_logs}

    def validation_step(self, *args):
        batch, batch_idx = args
        pred = self(batch)
        loss = self.loss_fn(pred, batch['rids'])
        pred = torch.sigmoid(pred)
        return {'val_loss': loss, 
                'pred': pred.detach().cpu().numpy(), 
                'label': batch['rids'].detach().cpu().numpy(),
               }


    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        pred = [oo  for o in outputs for oo in o['pred']]
        label = [oo for o in outputs for oo in o['label']]
        eval_f1, eval_T2, precision, recall, _, _ = self.f1_metric(label, pred)
        eval_f1 = torch.tensor(eval_f1, device=self.config.device)
        eval_T2 = torch.tensor(eval_T2, device=self.config.device)
        precision = torch.tensor(precision, device=self.config.device)
        recall = torch.tensor(recall, device=self.config.device)
        tensorboard_logs = {
            'val_loss': avg_loss, 
            'eval_f1': eval_f1
        }

        return {'progress_bar': tensorboard_logs, 
                'log': tensorboard_logs, 
                'val_loss': avg_loss, 
                'eval_f1': eval_f1, 
                'eval_T2': eval_T2, 
                'precision': precision, 
                'recall': recall,
               }


    def test_step(self, *args):
        batch, batch_idx, dl_idx = args
        pred = self(batch)
        loss = self.loss_fn(pred, batch['rids'])
        pred = torch.sigmoid(pred)
        return {'test_loss': loss, 
                'pred': pred.detach().cpu().numpy(), 
                'label': batch['rids'].detach().cpu().numpy(),
               }


    def test_epoch_end(self, outputs):
        for i, output in enumerate(outputs):
            if i == 0:
                dev_loss = torch.stack([x['test_loss'] for x in output]).mean()
                pred = [oo  for o in output for oo in o['pred']]
                label = [oo for o in output for oo in o['label']]
                dev_f1, dev_T2, precision, recall, best_pred, best_label = self.f1_metric(label, pred)
            elif i == 1:
                pred = [oo  for o in output for oo in o['pred']]
                dev_data = get_original_data(self.config.val_f)
                _, _, dev_f1c, dev_T2c = self.f1c_metric(pred, dev_data)
            elif i == 2:
                test_loss = torch.stack([x['test_loss'] for x in output]).mean()
                pred = [oo  for o in output for oo in o['pred']]
                label = [oo for o in output for oo in o['label']]
                test_f1, _, precision, recall, best_pred, best_label = self.f1_metric(label, pred, T2=dev_T2)
            elif i == 3:
                pred = [oo  for o in output for oo in o['pred']]
                tst_data = get_original_data(self.config.test_f)
                _, _, tst_f1c, _ = self.f1c_metric(pred, tst_data, T2=dev_T2c)

        tensorboard_logs = {
            'dev_loss': dev_loss,
            'test_loss': test_loss, 
            'dev_f1': dev_f1,
            'dev_f1c': dev_f1c,
            'test_f1': test_f1,
            'tst_f1c': tst_f1c,
            # 'test_T2': test_T2,
            # 'precision': precision,
            # 'recall': recall,
        }
        
        with open(os.path.join(self.logger.log_dir, 'output'), 'wb') as f:
            pickle.dump([best_pred, best_label], f) 
        return {'progress_bar': tensorboard_logs, 'log': tensorboard_logs}

    def train_dataloader(self):
        kwargs = dict(num_workers=self.config.num_workers, batch_size=self.config.batch_size, collate_fn=collate_fn, pin_memory=True)
        return data.DataLoader(self.ds_trn, shuffle=True, **kwargs) 

    def val_dataloader(self):
        kwargs = dict(shuffle=False, num_workers=self.config.num_workers, batch_size=32, collate_fn=collate_fn, pin_memory=True)
        return data.DataLoader(self.ds_val, **kwargs)

    def test_dataloader(self):
        kwargs = dict(shuffle=False, num_workers=self.config.num_workers, batch_size=32, collate_fn=collate_fn, pin_memory=True)
        dl_val = data.DataLoader(self.ds_val, **kwargs)
        dl_val_c = data.DataLoader(self.ds_val_c, **kwargs)
        dl_tst = data.DataLoader(self.ds_tst, **kwargs)
        dl_tst_c = data.DataLoader(self.ds_tst_c, **kwargs)
        return [dl_val, dl_val_c, dl_tst, dl_tst_c]
        
    @property
    def batch_size(self): return self.hparams.batch_size

    @batch_size.setter
    def batch_size(self, batch_size): self.hparams.batch_size = batch_size

    @property
    def lr(self): return self.hparams.lr

    @lr.setter
    def lr(self, lr): self.hparams.lr = lr
