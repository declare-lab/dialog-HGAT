import torch
from torch import nn
import os
import dgl
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning import callbacks
from pytorch_lightning import loggers

from utils.config import config
from model.trainer import Train_GraphDialogRe
from utils.data_reader import Vocab

if __name__ == "__main__":
    seed_everything(config.seed)
    dgl.random.seed(config.seed)
    
    model = Train_GraphDialogRe(config)

    logger = loggers.TensorBoardLogger(
        save_dir=config.save_dir
    )
    checkpoint_args = dict(
        monitor='eval_f1',
        mode='max',
    )
    early_stopping = callbacks.EarlyStopping(
        patience=5,
        strict=True,
        verbose=True,
        **checkpoint_args
    )
    ckpt_callback = callbacks.ModelCheckpoint(
        filepath=os.path.join(logger.log_dir, '{epoch}-{val_loss:.4f}-{eval_f1:.4f}-{eval_T2:.3f}'), # same path with logdir
        save_top_k=1,
        verbose=True,
        prefix='',
        **checkpoint_args,
    )
    
    trainer_args = dict(
        gpus=config.gpus, 
        num_nodes=config.num_nodes, 
        precision=config.precision, 
        early_stop_callback=False,  # early_stopping
        checkpoint_callback=ckpt_callback,
        logger=logger,
        limit_train_batches=1.0, 
        limit_val_batches=1.0,
        limit_test_batches=1.0,
        val_check_interval=1.0, 
        check_val_every_n_epoch=1,
        deterministic=True, # True,
        benchmark=False, # True,
        gradient_clip_val=5,
        profiler=True,
        progress_bar_refresh_rate=1,
        # auto_lr_find=True,
        # auto_scale_batch_size = 'bin', # None
        accumulate_grad_batches= config.actual_batch_size // config.batch_size,
    )
    
    trainer = Trainer(**trainer_args, resume_from_checkpoint=config.ckpt_path if config.ckpt_path else None)
    
    
    if config.mode == 'test':
        trainer.test(model)
    elif config.mode == 'train':
        trainer.fit(model)
        trainer.test()
