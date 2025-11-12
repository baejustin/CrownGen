import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from model.autoencoder import PVCNN2Base

class GaussianDiffusion:
    def __init__(self, betas, loss_type, model_mean_type, model_var_type):
        self.loss_type = loss_type
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        assert isinstance(betas, np.ndarray)
        self.np_betas = betas = betas.astype(np.float64)  # computations here in float64 for accuracy
        assert (betas > 0).all() and (betas <= 1).all()
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.from_numpy(np.cumprod(alphas, axis=0)).float()
        alphas_cumprod_prev = torch.from_numpy(np.append(1., alphas_cumprod[:-1])).float()

        self.betas = torch.from_numpy(betas).float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.alphas_cumprod_prev = alphas_cumprod_prev.float()

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).float()
        self.log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod).float()
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod).float()
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1).float()

        betas = torch.from_numpy(betas).float()
        alphas = torch.from_numpy(alphas).float()
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = posterior_variance
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(
            torch.max(posterior_variance, 1e-20 * torch.ones_like(posterior_variance)))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)
        self.posterior_mean_coef3 = (
            1.0 + ((torch.sqrt(self.alphas_cumprod) - 1.) * ( torch.sqrt(self.alphas_cumprod_prev) + torch.sqrt(alphas)))
            / (1.0 - self.alphas_cumprod))


    @staticmethod
    def _extract(a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        bs, = t.shape
        assert x_shape[0] == bs
        out = torch.gather(a, 0, t)
        assert out.shape == torch.Size([bs])

        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))

    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start
        variance = self._extract(1. - self.alphas_cumprod.to(x_start.device), t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape)

        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """ Diffuse the data (t == 0 means diffused for 1 step) """
        if noise is None:
            noise = torch.randn(x_start.shape, device=x_start.device)

        assert noise.shape == x_start.shape

        return (self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start +
                self._extract(self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape) * noise)

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """ Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0) """
        assert x_start.shape == x_t.shape
        posterior_mean = (self._extract(self.posterior_mean_coef1.to(x_start.device), t, x_t.shape) * x_start +
                          self._extract(self.posterior_mean_coef2.to(x_start.device), t, x_t.shape) * x_t)
        posterior_variance = self._extract(self.posterior_variance.to(x_start.device), t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped.to(x_start.device), t,
                                                       x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] ==
                x_start.shape[0])

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, denoise_fn, xt, model_kwargs, t, return_attn_weights: bool, return_pred_xstart: bool):

        model_output, saved_attns  = denoise_fn(xt, 
                                    t, 
                                    model_kwargs,
                                    return_attn_weights = return_attn_weights)

        device = xt.device
        shape = xt.shape

        if self.model_var_type in ['fixedsmall', 'fixedlarge']:  
            model_variance, model_log_variance = {
                'fixedlarge': (self.betas.to(device),
                               torch.log(torch.cat([self.posterior_variance[1:2], self.betas[1:]])).to(device)),
                'fixedsmall': (self.posterior_variance.to(device),
                               self.posterior_log_variance_clipped.to(device))}[self.model_var_type]

            model_variance = self._extract(model_variance, t, shape) * torch.ones_like(model_output)
            model_log_variance = self._extract(model_log_variance, t, shape) * torch.ones_like(model_output)

        else:
            raise NotImplementedError(self.model_var_type)

        if self.model_mean_type == 'eps':

            x_recon = self._predict_xstart_from_eps(xt, t=t, eps=model_output)

            model_mean, _, _ = self.q_posterior_mean_variance(x_start=x_recon, x_t=xt, t=t)

        else:
            raise NotImplementedError(self.loss_type)

        assert model_mean.shape == x_recon.shape
        assert model_variance.shape == model_log_variance.shape

        if return_pred_xstart:
            return model_mean, model_variance, model_log_variance, x_recon, saved_attns

        else:
            return model_mean, model_variance, model_log_variance, saved_attns

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape) * eps)

    ''' 
    ----- Sampling ----- 
    '''

    def p_sample(self, denoise_fn, xt, model_kwargs, t, noise_fn, return_attn_weights=False, return_pred_xstart=False):
        """ Sample from the model """

        model_mean, _, model_log_variance, pred_xstart, saved_attns = self.p_mean_variance(denoise_fn, xt=xt, model_kwargs=model_kwargs, t=t,
                                                                              return_attn_weights=return_attn_weights,
                                                                              return_pred_xstart=True)
        
        noise = noise_fn(size=model_mean.shape, dtype=model_mean.dtype, device=model_mean.device)

        nonzero_mask = torch.reshape(1 - (t == 0).float(), [xt.shape[0]] + [1] * (len(model_mean.shape) - 1))

        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

        return (sample, pred_xstart, saved_attns) if return_pred_xstart else (sample, saved_attns)

    def p_sample_loop(self, model_kwargs, denoise_fn,  noise_fn=torch.randn, return_attn_weights=False):

        x0 = model_kwargs['x0']
        B = x0.shape[0]
        device = x0.device
        img = noise_fn(size=x0.shape, dtype=torch.float, device=device)

        total_saved_attns = {}
        
        for t in tqdm(reversed(range(0, self.num_timesteps))):

            t_ = torch.empty(B, dtype=torch.int64, device=device).fill_(t)

            img, saved_attns = self.p_sample(denoise_fn=denoise_fn, xt=img, model_kwargs = model_kwargs, t=t_, noise_fn=noise_fn,
                                             return_attn_weights=return_attn_weights,return_pred_xstart=False)
            
            if return_attn_weights:
                total_saved_attns[t] = saved_attns

        return img, total_saved_attns


    ''' 
    ----- Losses ----- 
    '''

    def mse_mean_flat(self, B, noise, eps_recon, mask):
        total_loss = 0

        for b in range(B): # batch

            gt_noise = noise[b][mask[b].view(-1).bool()]
            pred_noise = eps_recon[b][mask[b].view(-1).bool()]
            loss_b = ((gt_noise - pred_noise)**2).mean()
            total_loss = total_loss + loss_b

        return total_loss / B

    def p_losses(self, denoise_fn, t, noise, model_kwargs):
        
        data_start = model_kwargs['x0']

        B = data_start.shape[0]
        assert t.shape == torch.Size([B])

        data_t = self.q_sample(x_start=data_start, t=t, noise=noise)

        if self.loss_type == 'mse':

            eps_recon, _  = denoise_fn(data_t, 
                                       t, 
                                       model_kwargs,
                                       return_attn_weights = False)
            
            losses = self.mse_mean_flat(B, noise, eps_recon, model_kwargs['l_mask'])

        else:
            raise NotImplementedError(self.loss_type)

        return losses



class PVCNN2(PVCNN2Base):

    num_n = 128 

    sa_blocks = [
        ((32, 2, 32), (512, 0.1, 32, (32, 64))),
        ((64, 3, 16), (256, 0.2, 32, (64, 128))),
        ((128, 3, 8), (64, 0.4, 32, (128, 256))),
        (None, (16, 0.8, 32, (256, 256, 512))),
    ]
    fp_blocks = [
        ((256, 256), (256, 3, 8)),
        ((256, 256), (256, 3, 8)),
        ((256, 128), (128, 2, 16)),
        ((128, 128, 64), (64, 2, 32)),
    ]


    def __init__(self, num_classes, embed_dim, use_att, dropout, extra_feature_channels=3,
                 width_multiplier=1.0, voxel_resolution_multiplier=1.0):
        super().__init__(num_classes=num_classes, embed_dim=embed_dim, use_att=use_att,
                         dropout=dropout, extra_feature_channels=extra_feature_channels,
                         width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier)

class Model(nn.Module):
    def __init__(self, cfg, betas, loss_type: str, model_mean_type: str, model_var_type: str,
                 width_mult: float, vox_res_mult: float):
        super(Model, self).__init__()

        self.diffusion = GaussianDiffusion(betas, loss_type, model_mean_type, model_var_type)

        self.model = PVCNN2(num_classes=cfg.model.nc, embed_dim=cfg.model.embed_dim,
                            use_att=cfg.model.attention, dropout=cfg.model.dropout, 
                            extra_feature_channels=cfg.model.extra_feature_nc,
                            width_multiplier=width_mult, voxel_resolution_multiplier=vox_res_mult)

    def _denoise(self, xt, t, model_kwargs, return_attn_weights=False):
        B = xt.shape[0]

        assert xt.dtype == torch.float
        assert t.shape == torch.Size([B]) and t.dtype == torch.int64

        out, attn_mask = self.model(xt, t,  return_attn_weights,**model_kwargs)

        return out, attn_mask


    def get_loss_iter_teethmask(self, noise_batch, model_kwargs):

        dentition_points = model_kwargs['x0']
        B = dentition_points.shape[0]
        t = torch.randint(0, self.diffusion.num_timesteps, size=(B,), device=noise_batch.device)
        
        losses = self.diffusion.p_losses(denoise_fn=self._denoise, 
                                         t=t, 
                                         noise=noise_batch,
                                         model_kwargs=model_kwargs)

        return losses

    def gen_samples(self, model_kwargs, noise_fn = torch.randn, return_attn_weights=False):

        out, total_saved_attns = self.diffusion.p_sample_loop(model_kwargs, self._denoise, noise_fn=noise_fn, 
                                                             return_attn_weights=return_attn_weights)

        return out, total_saved_attns

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def multi_gpu_wrapper(self, f):
        self.model = f(self.model)

def get_betas(schedule_type, b_start, b_end, time_num):
    if schedule_type == 'linear':
        betas = np.linspace(b_start, b_end, time_num)

    elif schedule_type == 'warm0.1':
        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.1)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)

    elif schedule_type == 'warm0.2':
        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.2)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)

    elif schedule_type == 'warm0.5':
        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.5)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)

    else:
        raise NotImplementedError(schedule_type)

    return betas