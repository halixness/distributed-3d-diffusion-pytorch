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
parser.add_argument('--train_data',type=str, default="./data/SRN/cars_train")
parser.add_argument('--val_data',type=str, default="")
args = parser.parse_args() 

if __name__ == '__main__':
    image_size = 64
    batch_size = 4
    data_workers = 4
            
    # Datasets and dataloader
    d = dataset('train', path=args.train_data, imgsize=image_size)
    d_val = dataset('val', path=args.val_data, imgsize=image_size)

    train_loader = MultiEpochsDataLoader(d, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=data_workers)
    val_loader = DataLoader(d_val, batch_size=128, shuffle=True, drop_last=True, num_workers=data_workers)

    model = Diff3D(
        pretrained_model = (None if args.transfer == "" else args.transfer),
        image_size = image_size,
        batch_size = batch_size,
    )

    trainer = pl.Trainer(gpus=1, precision=16, max_steps=100000)
    trainer.fit(model, train_loader)
    
"""
if args.transfer == "":
    now = './results/shapenet_SRN_car/'+str(int(time.time()))
    writer = SummaryWriter(now)
    step = 0
else:
    print('transfering from: ', args.transfer)
    
    ckpt = torch.load(os.path.join(args.transfer, 'latest.pt'))
    
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optim'])
    
    now = args.transfer
    writer = SummaryWriter(now)
    step = ckpt['step']


for e in range(100000):
    print(f'starting epoch {e}')
    
    lt = time.time()
    for img, R, T, K in tqdm(loader):
        
        warmup(optimizer=optimizer, step=step, last_step=10000000/batch_size, last_lr=1e-4)
        
        B = img.shape[0]
        
        optimizer.zero_grad()

        logsnr = logsnr_schedule_cosine(torch.rand((B,)))
        
        # Backward process computed loss
        loss = p_losses(model, img=img.cuda(), R=R.cuda(), T=T.cuda(), K=K.cuda(), logsnr=logsnr.cuda(), loss_type="l2", cond_prob=0.1)
        loss.backward()
        optimizer.step()
        
        writer.add_scalar("train/loss", loss.item(), global_step=step)
        writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], global_step=step)
        
        # Steps to eval, sample and print loss
        if step % 100 == 0:
            print("Saving model checkpoint at step: ", step)
            torch.save(model.state_dict(), "model.pt")    
        
        if step % 500 == 0:
            print("Loss:", loss.item())       
            
        # Val sampling step?
        if step % 1000 == 900: 
            model.eval()
            with torch.no_grad():
                for oriimg, R, T, K in loader_val:
                    
                    w = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).repeat(16)
                    img = sample(model, img=oriimg, R=R, T=T, K=K, w=w)

                    img = rearrange(((img[-1].clip(-1,1) + 1) * 127.5).astype(np.uint8), "(b a) c h w -> a c h (b w)", a=8, b=16)

                    gt = rearrange(((oriimg[:,1] + 1) * 127.5).detach().cpu().numpy().astype(np.uint8), "(b a) c h w -> a c h (b w)", a=8, b=16)
                    cd = rearrange(((oriimg[:,0] + 1) * 127.5).detach().cpu().numpy().astype(np.uint8), "(b a) c h w -> a c h (b w)", a=8, b=16)

                    fi = np.concatenate([cd, gt, img], axis=2)
                    for i, ww in enumerate([0, 1, 2, 3, 4, 5, 6, 7]):
                        writer.add_image(f"train/{ww}", fi[i], step)
                    break

            print('image sampled!')
            writer.flush()
            model.train()
            
        
        if step == int(10000000/batch_size):
            torch.save({'optim':optimizer.state_dict(), 'model':model.state_dict(), 'step':step}, now+f"/after_warmup.pt")
        
        step += 1
        starttime = time.time()
        
    
    if e%20 == 0:
        torch.save({'optim':optimizer.state_dict(), 'model':model.state_dict(), 'step':step, 'epoch':e}, now+f"/latest.pt")
 
"""