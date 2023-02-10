"""
    https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
    https://gist.github.com/sgraaf/5b0caa3a320f28c27c12b5efeb35aa4c
"""

import os
import sys
import tempfile
import numpy as np
from tqdm import tqdm
import time
import argparse

from einops import rearrange
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from xunet import XUNet
from SRNdataset import dataset, MultiEpochsDataLoader

#from tensorboardX import SummaryWriter

# --------------------------------------------------------------------------

def logsnr_schedule_cosine(t, *, logsnr_min=-20., logsnr_max=20.):
    b = np.arctan(np.exp(-.5 * logsnr_max))
    a = np.arctan(np.exp(-.5 * logsnr_min)) - b

    return -2. * torch.log(torch.tan(a * t + b))

def xt2batch(x, logsnr, z, R, T, K):
    b = x.shape[0]

    return {
        'x': x.cuda(),
        'z': z.cuda(),
        'logsnr': torch.stack([logsnr_schedule_cosine(torch.zeros_like(logsnr)), logsnr], dim=1).cuda(),
        'R': R.cuda(),
        't': T.cuda(),
        'K':K.cuda(),
    }

# ----------------------- Diffusion forward process -----------------------

def q_sample(z, logsnr, noise):
    """
        Forward: q(x_t|x_0)
    """ 
    alpha = logsnr.sigmoid().sqrt()
    sigma = (-logsnr).sigmoid().sqrt()
    
    alpha = alpha[:,None, None, None]
    sigma = sigma[:,None, None, None]

    return alpha * z + sigma * noise

@torch.no_grad()
def sample(model, img, R, T, K, w, timesteps=256):
    """
        Forward process
    """
    x = img[:, 0]
    img = torch.randn_like(x)
    imgs = []
    
    logsnrs = logsnr_schedule_cosine(torch.linspace(1., 0., timesteps+1)[:-1])
    logsnr_nexts = logsnr_schedule_cosine(torch.linspace(1., 0., timesteps+1)[1:])
    
    for logsnr, logsnr_next in tqdm(zip(logsnrs, logsnr_nexts)): # [1, ..., 0] = size is 257
        img = p_sample(model, x=x, z=img, R=R, T=T, K=K, logsnr=logsnr, logsnr_next=logsnr_next, w=w)
        imgs.append(img.cpu().numpy())

    return imgs

def p_losses(denoise_model, img, R, T, K, logsnr, noise=None, loss_type="l2", cond_prob=0.1):
    """
        Loss function (L2 default)
    """
    B = img.shape[0]
    x = img[:, 0]
    z = img[:, 1]

    # Epsilon
    if noise is None:
        noise = torch.randn_like(x)

    # Forward noising process
    z_noisy = q_sample(z=z, logsnr=logsnr, noise=noise)
    
    cond_mask = (torch.rand((B,)) > cond_prob).cuda()
    x_condition = torch.where(cond_mask[:, None, None, None], x, torch.randn_like(x))
    batch = xt2batch(x=x_condition, logsnr=logsnr, z=z_noisy, R=R, T=T, K=K)
    
    # Denoising (predicts noise)
    predicted_noise = denoise_model(batch, cond_mask=cond_mask.cuda())

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)

    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    
    else:
        raise NotImplementedError()

    return loss

# ----------------------- Diffusion backward process -----------------------

@torch.no_grad()
def p_sample(model, x, z, R, T, K, logsnr, logsnr_next, w):
    """
        Backward process: epsilon_0(x_t, t)
    """
    model_mean, model_variance = p_mean_variance(model, x=x, z=z, R=R, T=T, K=K, logsnr=logsnr, logsnr_next=logsnr_next, w=w)
    
    if logsnr_next==0:
        return model_mean
    
    return model_mean + model_variance.sqrt() * torch.randn_like(x).cpu()


@torch.no_grad()
def p_mean_variance(model, x, z, R, T, K, logsnr, logsnr_next, w=2.0):
    """
        Backward process (and variance)
    """
    strt = time.time()
    b = x.shape[0]
    w = w[:, None, None, None]
    
    c = - torch.special.expm1(logsnr - logsnr_next)
    
    squared_alpha, squared_alpha_next = logsnr.sigmoid(), logsnr_next.sigmoid()
    squared_sigma, squared_sigma_next = (-logsnr).sigmoid(), (-logsnr_next).sigmoid()
    
    alpha, sigma, alpha_next = map(lambda x: x.sqrt(), (squared_alpha, squared_sigma, squared_alpha_next))
    
    # batch = xt2batch(x, logsnr.repeat(b), z, R)
    batch = xt2batch(x, logsnr.repeat(b), z, R, T, K)
    
    strt = time.time()
    pred_noise = model(batch, cond_mask= torch.tensor([True]*b)).detach().cpu()
    batch['x'] = torch.randn_like(x).cuda()
    pred_noise_unconditioned = model(batch, cond_mask= torch.tensor([False]*b)).detach().cpu()
    
    pred_noise_final = (1+w) * pred_noise - w * pred_noise_unconditioned
    
    z = z.detach().cpu()
    
    z_start = (z - sigma * pred_noise_final) / alpha
    z_start.clamp_(-1., 1.)
    
    model_mean = alpha_next * (z * (1 - c) / alpha + c * z_start)
    
    posterior_variance = squared_sigma_next * c
    
    return model_mean, posterior_variance


def warmup(optimizer, step, last_step, last_lr):
    """
        Step-wise learning rate update
    """
    if step < last_step:
        optimizer.param_groups[0]['lr'] = step / last_step * last_lr
        
    else:
        optimizer.param_groups[0]['lr'] = last_lr
        
# --------------------------------------------------------------------------

def setup(rank, world_size):
    print(f"======== World size: {world_size}")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def run(fn, transfer, world_size):
    mp.spawn(fn,
             args=(world_size, transfer),
             nprocs=world_size,
             join=True)

def cleanup():
    dist.destroy_process_group()

# -----------------------------------

def train(rank, world_size, transfer=""):
    """
        Possible parallelism:
        - Epoch wise: n_batches/k epochs per device
        - Data wise: n_batches/k batches per device, per x epochs
    """
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # ------------ Init
    step = 0
    num_epochs = 10
    image_size = 64
    batch_size = 128
    steps_plot_loss = 50
    steps_ckpt = 50
    n_workers = 16
    epochs_plot_loss = steps_plot_loss

    # ------------ Data loading
    # d_val = dataset('val', path='./data/SRN/cars_train', imgsize=image_size)
    # loader_val = DataLoader(d_val, batch_size=128, shuffle=True, drop_last=True, num_workers=16)

    d = dataset('train', path='./data/SRN/cars_train', imgsize=image_size)
    d = DistributedSampler(d)
    
    loader = MultiEpochsDataLoader(d, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=n_workers)
    
    # Model setting
    model = XUNet(H=image_size, W=image_size, ch=128).to(rank)
    ddp_model = DDP(
        model,
        device_ids=[rank]
    )

    optimizer = Adam(ddp_model.parameters(), lr=1e-4, betas=(0.9, 0.99))
    
    # Load saved model if defined
    if transfer == "":
        step = 0
    else:
        print('transfering from: ', transfer)
        
        # Mapped ckpt loading
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}

        ckpt = torch.load(os.path.join(transfer, 'latest.pt'), map_location=map_location)
        ddp_model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optim'])
        
        now = transfer
        step = ckpt['step']
        
    # Training loop
    for e in range(num_epochs):

        print(f'starting epoch {e}')
        
        ddp_model.train()
        dist.barrier()
        
        lt = time.time()

        # For each sample in the dataset
        for img, R, T, K in tqdm(loader):
            
            # Learning rate compute
            warmup(optimizer, step, num_epochs/batch_size, 0.0001)
            
            optimizer.zero_grad()

            B = img.shape[0]
            logsnr = logsnr_schedule_cosine(torch.rand((B,)))
            
            # Forward and loss compute
            loss = p_losses(model, img=img.cuda(), R=R.cuda(), T=T.cuda(), K=K.cuda(), logsnr=logsnr.cuda(), loss_type="l2", cond_prob=0.1)
            loss.backward()
            
            optimizer.step()
            
            # writer.add_scalar("train/loss", loss.item(), global_step=step)
            # writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], global_step=step)
            
            # Plot loss
            if step % steps_ckpt == 0:
                print("Loss:", loss.item())

            # Save checkpoint (ONLY FOR RANK 0)
            if rank == 0 and step % steps_plot_loss == 0:
                torch.save({'optim':optimizer.state_dict(), 'model':model.state_dict(), 'step':step}, now+f"/after_warmup.pt")

            # Synchronization point            
            dist.barrier()
            step += 1
            

        # Epoch checkpoint save (ONLY FOR RANK 0)
        if rank == 0 and e % epochs_plot_loss == 0:
            torch.save({'optim':optimizer.state_dict(), 'model':model.state_dict(), 'step':step, 'epoch':e}, now+f"/latest.pt")
        
        # Synchronization point            
        dist.barrier()
    
    cleanup()

# -----------------------------------

if __name__ == "__main__":
    MIN_GPUS = 1

    parser = argparse.ArgumentParser()
    parser.add_argument('--transfer',type=str, default="")
    args = parser.parse_args()

    n_gpus = torch.cuda.device_count()
    
    assert n_gpus >= MIN_GPUS, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    
    run(train, args.transfer, world_size)
    