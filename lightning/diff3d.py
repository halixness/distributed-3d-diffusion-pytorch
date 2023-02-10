import pytorch_lightning as pl
import numpy as np
import torch
from torch.optim import Adam
import numpy as np
import torch.nn.functional as F
from xunet import XUNet

class Diff3D(pl.LightningModule):
    """
        • We use a learning rate with peak value 0.0001, using linear warmup for the first 10 million examples (where one batch has batch_size examples), following Karras et al. (2022).
        • We use a global batch size of 128.
        • We train each batch element as an unconditional example 10% of the time to enable classifier-free
        guidance. This is done by overriding the conditioning frame to be at the maximum noise level.
        We note other options are possible (e.g., zeroing-out the conditioning frame), but we chose the
        option which is most compatible with our neural architecture.
        • We use the Adam optimizer (Kingma & Ba, 2014) with β1 = 0.9 and β2 = 0.99.
        • We use EMA decay for the model parameters, with a half life of 500K examples (where one batch
        has batch_size examples) following Karras et al. (2022).
    """

    # -----------------
    def __init__(self, pretrained_model=None, n_samples=10000000, image_size = 64, batch_size = 128, lr=1e-4, use_scheduler=False):
        super().__init__()
        """
            pretrained_model: String    path to the .pt file    
        """
        self.n_samples = n_samples # used in implementation for warmup???
        self.batch_size = batch_size
        self.image_size = image_size
        self.lr = lr
        self.use_scheduler = use_scheduler
        self.step = 0

        self.pretrained_optim = None
        self.optimizer = None
        
        self.xunet_denoiser = XUNet(H=self.image_size, W=self.image_size, ch=128) 
        
        if pretrained_model is not None:
            print('[-] Loading pre-trained model: ', pretrained_model)
            ckpt = torch.load(pretrained_model)
            
            self.xunet_denoiser.load_state_dict(ckpt['model'])
            self.pretrained_optim = ckpt['optim']
            
            #now = args.transfer
            #writer = SummaryWriter(now)
            #step = ckpt['step']

    # -----------------
    def forward(self, in_x, noise=None, cond_prob=0.1):

        img, R, T, K = in_x
        
        # Diffusion denoising steps
        B = img.shape[0]
        x = img[:, 0]
        z = img[:, 1]

        logsnr = self.logsnr_schedule_cosine(torch.rand((B,)))
        
        if noise is None:
            noise = torch.randn_like(x)

        z_noisy = self.q_sample(z=z, logsnr=logsnr, noise=noise)
        
        cond_mask = torch.Tensor(torch.rand((B,)) > cond_prob).to(self.device)
        x_condition = torch.where(cond_mask[:, None, None, None], x, torch.randn_like(x).to(self.device))
        
        batch = self.xt2batch(x=x_condition, logsnr=logsnr, z=z_noisy, R=R, T=T, K=K)
        predicted_noise = self.xunet_denoiser(batch, cond_mask=cond_mask).to(self.device)

        return predicted_noise

    # -----------------
    def training_step(self, batch, batch_idx, loss_type="l2"):

        if not self.use_scheduler:
            self.warmup()
        
        self.optimizers().zero_grad()
    
        noise = torch.randn_like(batch[0][:, 0]) # batch => img => x
        predicted_noise = self.forward(in_x=batch, noise=noise)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        self.step += 1

        if self.use_scheduler:
            self.lr_schedulers().step()

        return loss

    # -----------------
    def configure_optimizers(self):
        
        optimizer = Adam(self.xunet_denoiser.parameters(), lr=self.lr, betas=(0.9, 0.99))
        
        if self.pretrained_optim is not None:
            optimizer.load_state_dict(self.pretrained_optim)

        if self.use_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    # -----------------
    def warmup(self):
        """
            Warmup phase, gradient setting
        """
        last_step = self.n_samples/self.batch_size, 

        if self.step < last_step[0]:
            self.optimizers().param_groups[0]['lr'] = (self.step / last_step[0]) * self.lr
        else:
            self.optimizers().param_groups[0]['lr'] = self.lr

    # ---------------------------------------------------------------------------

    def logsnr_schedule_cosine(self, t, *, logsnr_min=-20., logsnr_max=20.):
        b = np.arctan(np.exp(-.5 * logsnr_max))
        a = np.arctan(np.exp(-.5 * logsnr_min)) - b
        
        return -2. * torch.log(torch.tan(a * t + b))

    def xt2batch(self, x, logsnr, z, R, T, K):
        b = x.shape[0]

        return {
            'x': x,
            'z': z,
            'logsnr': torch.stack([self.logsnr_schedule_cosine(torch.zeros_like(logsnr)), logsnr], dim=1),
            'R': R,
            't': T,
            'K':K,
        }

    # ----------------------- Diffusion forward process -----------------------

    @torch.no_grad()
    def sample(self, model, img, R, T, K, w, timesteps=256):
        """
            Forward process
        """
        x = img[:, 0]
        img = torch.randn_like(x)
        imgs = []
        
        logsnrs = self.logsnr_schedule_cosine(torch.linspace(1., 0., timesteps+1)[:-1])
        logsnr_nexts = self.logsnr_schedule_cosine(torch.linspace(1., 0., timesteps+1)[1:])
        
        for logsnr, logsnr_next in tqdm(zip(logsnrs, logsnr_nexts)): # [1, ..., 0] = size is 257
            img = self.p_sample(model, x=x, z=img, R=R, T=T, K=K, logsnr=logsnr, logsnr_next=logsnr_next, w=w)
            imgs.append(img.cpu().numpy())
        return imgs

    def q_sample(self, z, logsnr, noise):
        """
            Forward: q(x_t|x_0)
        """
        # lambdas = self.logsnr_schedule_cosine(t)
        
        alpha = torch.Tensor(logsnr.sigmoid().sqrt()).type_as(z) 
        sigma = torch.Tensor((-logsnr).sigmoid().sqrt()).type_as(z)
        
        alpha = alpha[:,None, None, None]
        sigma = sigma[:,None, None, None]

        return alpha * z + sigma * noise


    # ----------------------- Diffusion backward process -----------------------

    @torch.no_grad()
    def p_sample(self, model, x, z, R, T, K, logsnr, logsnr_next, w):
        """
            Backward process: epsilon_0(x_t, t)
        """
        
        model_mean, model_variance = self.p_mean_variance(model, x=x, z=z, R=R, T=T, K=K, logsnr=logsnr, logsnr_next=logsnr_next, w=w)
        
        if logsnr_next==0:
            return model_mean
        
        return model_mean + model_variance.sqrt() * torch.randn_like(x).cpu()


    @torch.no_grad()
    def p_mean_variance(self, model, x, z, R, T, K, logsnr, logsnr_next, w=2.0):
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
        
        # batch = self.xt2batch(x, logsnr.repeat(b), z, R)
        batch = self.xt2batch(x, logsnr.repeat(b), z, R, T, K)
        
        strt = time.time()

        # Predicted noise
        pred_noise = model(batch, cond_mask= torch.tensor([True]*b)).detach().cpu()
        batch['x'] = torch.randn_like(x)
        pred_noise_unconditioned = model(batch, cond_mask= torch.tensor([False]*b)).detach().cpu()
        
        pred_noise_final = (1+w) * pred_noise - w * pred_noise_unconditioned
        
        z = z.detach().cpu()
        
        # actual predicted x_0
        z_start = (z - sigma * pred_noise_final) / alpha
        z_start.clamp_(-1., 1.)
        
        model_mean = alpha_next * (z * (1 - c) / alpha + c * z_start)
        
        posterior_variance = squared_sigma_next * c
        
        return model_mean, posterior_variance
        
    # ---------------------------------------------------------------------------


