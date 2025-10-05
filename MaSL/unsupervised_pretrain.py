import os
import argparse
import pickle
# import pdb

import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from model import UnsupervisedPretrain
from utils import UnsupervisedPretrainLoader, collate_fn_unsupervised_pretrain
torch.cuda.empty_cache()
torch.backends.cudnn.enabled = False

     
class LitModel_supervised_pretrain(pl.LightningModule):
    def __init__(self, args, save_path):
        super().__init__()
        self.args = args
        self.save_path = save_path
        self.T = 0.2
        self.model = UnsupervisedPretrain(emb_size=256, heads=8, depth=4, dropout=args.dropout) 
        
    def training_step(self, batch, batch_idx):

        noaug_combined_tu, aug_combined_tu,noaug_combined_shhs,aug_combined_shhs = batch
        shhs_samples = (noaug_combined_shhs,aug_combined_shhs)
        tu_samples = (noaug_combined_tu, aug_combined_tu)
        
        contrastive_loss = 0
        if len(noaug_combined_shhs) > 0:
            """
            For shhs
            """
            shhs_masked_emb, shhs_samples_emb = self.model(shhs_samples, 0)

            # L2 normalize
            shhs_samples_emb = F.normalize(shhs_samples_emb, dim=1, p=2)
            shhs_masked_emb = F.normalize(shhs_masked_emb, dim=1, p=2)
            N = shhs_samples_emb.shape[0]

            # representation similarity matrix, NxN
            logits = torch.mm(shhs_samples_emb, shhs_masked_emb.t()) / self.T
            labels = torch.arange(N).to(logits.device)
            contrastive_loss += F.cross_entropy(logits, labels, reduction="mean")

        """
        For tu
        """
        tu_masked_emb, tu_samples_emb = self.model(tu_samples, 0) # origin:2

        # For shhs
        tu_samples_emb = F.normalize(tu_samples_emb, dim=1, p=2)
        tu_masked_emb = F.normalize(tu_masked_emb, dim=1, p=2)
        N = tu_samples_emb.shape[0]

        # representation similarity matrix, NxN
        logits = torch.mm(tu_samples_emb, tu_masked_emb.t()) / self.T
        labels = torch.arange(N).to(logits.device)
        contrastive_loss += F.cross_entropy(logits, labels, reduction="mean")

        self.log("train_loss", contrastive_loss)
        # writer.add_scalar("train_loss", contrastive_loss, self.global_step)
        return contrastive_loss
    
    def on_train_epoch_end(self):
        # Clear the list before the start of a new epoch
        if (self.current_epoch + 1) % 10 == 0:
            print(f"SAVE GLOBAL STEP {self.global_step} in {self.current_epoch} to {self.save_path}")
            self.trainer.save_checkpoint(
                filepath=f"{self.save_path}/epoch={self.current_epoch}_step={self.global_step}.ckpt"
            )


    def configure_optimizers(self):
        # set optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )

        # set learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=14310, gamma=0.3
        )

        return [optimizer], [scheduler]

    
def prepare_dataloader(args):
    # set random seed
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # define the (seizure) data loader
    root_shhs = args.shhs_path
    root_TUH = args.TUHseries_path
    # loader = UnsupervisedPretrainLoader(root_prest, root_shhs)
    loader = UnsupervisedPretrainLoader(root_shhs,root_TUH)
    train_loader = torch.utils.data.DataLoader(
        loader,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=True,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_fn_unsupervised_pretrain,
    )
    
    return train_loader
 
 
def pretrain(args):
    
    # get data loaders
    train_loader = prepare_dataloader(args)
    
    # define the model
    model = LitModel_supervised_pretrain(args, args.save_path)
    
   
    logger_csv = CSVLogger('/home/replace/EEG/code/MaSL/csv_logs', name='unsup_model',flush_logs_every_n_steps=100)
    
    trainer = pl.Trainer(
        devices=[0],
        accelerator="gpu",
        benchmark=True,
        enable_checkpointing=True,
        logger=logger_csv,
        max_epochs=args.epochs+1,
    )

    # train the model
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1, help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
    parser.add_argument("--num_workers", type=int, default=32, help="number of workers")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    # dataset parameters    
    parser.add_argument('--shhs_path', type=str, nargs='?', const=None, default="/home/replace/EEG/data/SHHS/processed",help='Path to SHHS data root directory or "None" to indicate no path.')
    parser.add_argument('--TUHseries_path', type=str, nargs='?', const=None, default="/mnt/replace_disk/EEG_data/TUHseries",help='Path to TUAB data root directory or "None" to indicate no path.')
    
    # model parameters
    parser.add_argument("--n_channels", type=int, default=2, help="number of channels in pretrained model") # 2-shhs

    # training parameters
    parser.add_argument("--save_path", type=str, default="./log-pretrain/unsupervised/tu_checkpoints_mamba", help="checkpoint path")


    args = parser.parse_args()
    print (args)

    pretrain(args)
    
    
    