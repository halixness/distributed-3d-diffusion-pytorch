import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from einops import rearrange
import time

from SRNdataset import dataset, MultiEpochsDataLoader
from tensorboardX import SummaryWriter
import os
import argparse
from diff3d import Diff3D

# Arg parsing and management
parser = argparse.ArgumentParser()
parser.add_argument('--transfer',type=str, default="")
parser.add_argument('--train_data',type=str, default=os.path.join("data", "SRN", "cars_train"))
parser.add_argument('--val_data',type=str, default="")
args = parser.parse_args() 

if __name__ == '__main__':
    image_size = 64
    batch_size = 4
    data_workers = 4
            
    # Datasets and dataloader
    d = dataset('train', path=args.train_data, picklefile=os.path.join(args.train_data, "cars.pickle"), imgsize=image_size)
    #d_val = dataset('val', path=args.val_data, imgsize=image_size)

    train_loader = MultiEpochsDataLoader(d, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=data_workers)
    #val_loader = DataLoader(d_val, batch_size=128, shuffle=True, drop_last=True, num_workers=data_workers)

    model = Diff3D(
        n_samples=len(train_loader.dataset),
        use_scheduler=False,
        pretrained_model = (None if args.transfer == "" else args.transfer),
        image_size = image_size,
        batch_size = batch_size,
    )

    trainer = pl.Trainer(accelerator="gpu", devices=1, precision=16, max_steps=100000)
    trainer.fit(model, train_loader)
    