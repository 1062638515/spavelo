import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.distributions import Categorical, Dirichlet, MixtureSameFamily, Normal
from torch.distributions import kl_divergence
import numpy as np
import pandas as pd
from SVGP import SVGP
from I_PID import PIDControl
from VAE_utils import *
from collections import deque
import time
from VAE_utils import NegativeBinomial
import scanpy as sc
import scvelo as scv
from scvelo.tools._em_model_utils import compute_shared_time, root_time
from scvelo.tools._em_model_core import get_connectivities
from scvelo.tools.velocity_pseudotime import velocity_pseudotime
from scvelo.tools.utils import scale
from scvelo import logging as logg


class EarlyStopping:
    """Early stops the training if loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, modelfile="model.pt"):
        """
        Args:
            patience (int): How long to wait after last time loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.Inf
        self.model_file = modelfile
        print(modelfile)

    def __call__(self, loss, model):
        if np.isnan(loss):
            self.early_stop = True
        score = -loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                model.load_model(self.model_file)
        else:
            self.best_score = score
            self.save_checkpoint(loss, model)
            self.counter = 0

    def save_checkpoint(self, loss, model):
        '''Saves model when loss decrease.'''
        if self.verbose:
            print(f'Loss decreased ({self.loss_min:.6f} --> {loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.model_file)
        self.loss_min = loss


class KineticRecoder:
    def __init__(self, result_dir, gene):
        self.alpha = []
        self.beta = []
        self.gamma = []
        self.epoch = []
        self.gene = gene
        self.result_dir = result_dir

    def update(self, rates, epoch=None):
        self.alpha.append(rates["alpha"])
        self.beta.append(rates["beta"])
        self.gamma.append(rates["gamma"])
        if epoch is None:
            self.epoch.append(len(self.alpha))
        self.epoch.append(epoch)

    def save(self):
        alpha = pd.DataFrame(self.alpha, columns=self.gene)
        beta = pd.DataFrame(self.beta, columns=self.gene)
        gamma = pd.DataFrame(self.gamma, columns=self.gene)
        alpha["epoch"] = self.epoch
        beta["epoch"] = self.epoch
        gamma["epoch"] = self.epoch
        alpha.to_csv(f"{self.result_dir}/alpha.csv", index=False)
        beta.to_csv(f"{self.result_dir}/beta.csv", index=False)
        gamma.to_csv(f"{self.result_dir}/gamma.csv", index=False)


class SPAVELO(nn.Module):
    def __init__(self, n_genes, true_time_switch, t_max, penalty_scale, log_variational, GP_dim, Normal_dim, encoder_layers, decoder_layers, encoder_dropout, decoder_dropout, 
                 fixed_inducing_points, initial_inducing_points, fixed_gp_params, kernel_scale, N_train, KL_loss, dynamicVAE, init_beta, min_beta, max_beta, KL_weight,
                 gamma_unconstr_init, alpha_unconstr_init, alpha_1_unconstr_init, lambda_alpha_unconstr_init, switch_spliced, switch_unspliced, linear_decoder, 
                 time_dep_transcription_rate, dirichlet_concentration, dtype, device, use_raw, adata=None, spliced_layer="spliced", unspliced_layer="unspliced",
                 task="standard", use_rho=True):
        super(SPAVELO, self).__init__()
        torch.set_default_dtype(dtype)
        self.svgp = SVGP(fixed_inducing_points=fixed_inducing_points, initial_inducing_points=initial_inducing_points,
                fixed_gp_params=fixed_gp_params, kernel_scale=kernel_scale, jitter=1e-8, N_train=N_train, dtype=dtype, device=device, task=task)
        self.n_genes = n_genes
        self.t_max = t_max
        self.penalty_scale = penalty_scale
        self.log_variational = log_variational
        self.PID = PIDControl(Kp=0.01, Ki=-0.005, init_beta=init_beta, min_beta=min_beta, max_beta=max_beta)
        self.KL_loss = KL_loss          # expected KL loss value
        self.dynamicVAE = dynamicVAE
        self.KL_beta = init_beta           # beta controls the weight of reconstruction loss
        self.KL_weight = KL_weight
        self.dtype = dtype
        self.GP_dim = GP_dim            # dimension of latent Gaussian process embedding
        self.Normal_dim = Normal_dim    # dimension of latent standard Gaussian embedding
        self.device = device
        n_input = n_genes*2
        latent_dim = GP_dim+Normal_dim
        self.encoder = DenseEncoder(input_dim=n_input, hidden_dims=encoder_layers, latent_dim=latent_dim, activation="elu", dropout=encoder_dropout, norm="batchnorm", dtype=dtype)

        if switch_spliced is not None:
            self.register_buffer("switch_spliced", torch.from_numpy(switch_spliced))
        else:
            self.switch_spliced = None
        if switch_unspliced is not None:
            self.register_buffer("switch_unspliced", torch.from_numpy(switch_unspliced))
        else:
            self.switch_unspliced = None

        self.switch_time_unconstr = torch.nn.Parameter(7 + 0.5 * torch.randn(n_genes), requires_grad=True)
        if true_time_switch is not None:
            self.register_buffer("true_time_switch", torch.from_numpy(true_time_switch))
        else:
            self.true_time_switch = None


        # degradation
        if gamma_unconstr_init is None:
            self.gamma_unconstr = nn.Parameter(0.5 * torch.ones(n_genes), requires_grad=True)
        else:
            self.gamma_unconstr = nn.Parameter(torch.from_numpy(gamma_unconstr_init), requires_grad=True)

        # splicing
        # first samples around 1
        self.beta_unconstr = nn.Parameter(torch.ones(n_genes), requires_grad=True)

        # transcription
        if alpha_unconstr_init is None:
            self.alpha_unconstr = nn.Parameter(F.softplus(torch.zeros(n_genes)), requires_grad=True)
        else:
            self.alpha_unconstr = nn.Parameter(F.softplus(torch.from_numpy(alpha_unconstr_init)), requires_grad=True)

        self.time_dep_transcription_rate = time_dep_transcription_rate

        if alpha_1_unconstr_init is None:
            self.alpha_1_unconstr = nn.Parameter(torch.zeros(n_genes), requires_grad=True)
        else:
            self.alpha_1_unconstr = nn.Parameter(torch.from_numpy(alpha_1_unconstr_init), requires_grad=True)

        if lambda_alpha_unconstr_init is None:
            self.lambda_alpha_unconstr = nn.Parameter(torch.zeros(n_genes), requires_grad=True)
        else:
            self.lambda_alpha_unconstr = nn.Parameter(torch.from_numpy(lambda_alpha_unconstr_init), requires_grad=True)
        self.lambda_alpha_unconstr.requires_grad = time_dep_transcription_rate

        self.use_raw = use_raw
        if self.use_raw:
            self.scale_unconstr = nn.Parameter(0.5 * torch.ones(n_input), requires_grad=True)
        else:
            self.scale_unconstr = nn.Parameter(-1 * torch.ones(n_input), requires_grad=True)

        # Initialize decoder with StateDecoder option
        gamma, beta, alpha, alpha_1, lambda_alpha = self._get_rates()
        self.decoder = SpatialVeloDecoder(latent_dim=latent_dim, hidden_dims=decoder_layers, output_dim=n_genes, 
                                          linear_decoder=linear_decoder, activation="elu", dropout=decoder_dropout, 
                                          norm="batchnorm", dtype=dtype)

        self.alpha_1_unconstr.requires_grad = time_dep_transcription_rate
        self.u_max = torch.inf
        self.u_min = 0
        self.s_max = torch.inf
        self.s_min = 0

        self.dirichlet_concentration = dirichlet_concentration
        self.kinetic_recorder = None
        self.use_rho = use_rho
        self.to(device)

    def save_model(self, path):
        torch.save(self.state_dict(), path)


    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)


    def forward(self, x, y_spliced, y_unspliced, epoch, n_samples=1, latent_dim=None, prior_prob=None, gamma_r2=None):
        """
        Forward pass.

        Parameters:
        -----------
        x: mini-batch of positions.
        y_spliced: mini-batch of spliced counts.
        y_unspliced: mini-batch of unspliced counts.
        n_samples: number of samplings of the posterior distribution of latent embedding.
        """ 

#        self.train()

        inference_outputs = self.encoder_inference(x, y_spliced, y_unspliced)
        kl_gp = inference_outputs["kl_gp"]
        kl_gaussian = inference_outputs["kl_gaussian"]
        latent_dist = inference_outputs["latent_dist"]
        p_m = inference_outputs["p_m"]
        p_v = inference_outputs["p_v"]

        latent_samples = []
        for _ in range(n_samples):
            latent_samples_ = latent_dist.rsample()
            latent_samples.append(latent_samples_)

        gamma, beta, alpha, alpha_1, lambda_alpha = self._get_rates()
        reconst_loss = 0
        end_penalty_loss = 0
        
        # 使用gamma_r2创建mask，判断哪些基因使用dynamo prior，哪些使用标准prior
        if gamma_r2 is not None:
            gamma_r2_expanded = gamma_r2.reshape(1, -1) * torch.ones(size=(x.size(0), y_spliced.size(1)), device=self.device)
            dynamo_mask = torch.where(gamma_r2_expanded > 0, 1, 0)
            standard_mask = torch.where(dynamo_mask > 0, 0, 1)
        else:
            # 如果没有提供gamma_r2，则全部使用标准prior
            dynamo_mask = torch.zeros(size=(x.size(0), y_spliced.size(1)), device=self.device)
            standard_mask = torch.ones(size=(x.size(0), y_spliced.size(1)), device=self.device)
        
        for z in latent_samples:
            generative_outputs = self.decoder_generative(z, gamma, beta, alpha, alpha_1, lambda_alpha, latent_dim=latent_dim, y_spliced=y_spliced, y_unspliced=y_unspliced)

            px_pi = generative_outputs["px_pi"]
            px_pi_alpha = generative_outputs["px_pi_alpha"]

            end_penalty = generative_outputs["end_penalty"]
            mixture_dist_s = generative_outputs["mixture_dist_s"]
            mixture_dist_u = generative_outputs["mixture_dist_u"]

            end_penalty_loss += end_penalty

            reconst_loss_s = -mixture_dist_s.log_prob(y_spliced)
            reconst_loss_u = -mixture_dist_u.log_prob(y_unspliced)

            reconst_loss += reconst_loss_u.sum() + reconst_loss_s.sum()

            # Use provided prior probabilities for Dirichlet KL divergence
            if prior_prob is None:
                raise ValueError("prior_prob must be provided when use_state_decoder=False")

            # 根据gamma_r2 mask使用不同的prior
            # dynamo_mask > 0的基因使用dynamo的prior，其他使用标准的0.25 uniform prior
            kl_pi_dynamo = kl_divergence(Dirichlet(px_pi_alpha), Dirichlet(torch.clamp(prior_prob, 1e-6, None))) * dynamo_mask
            kl_pi_standard = kl_divergence(Dirichlet(px_pi_alpha), Dirichlet(self.dirichlet_concentration * torch.ones_like(prior_prob, device=self.device))) * standard_mask
            kl_pi = (self.KL_weight * kl_pi_dynamo + kl_pi_standard).sum() / px_pi_alpha.shape[1]
        reconst_loss = reconst_loss / n_samples
        end_penalty_loss = end_penalty_loss / n_samples
        kl_pi = kl_pi / n_samples

        # ELBO
        penalty_weight = self._compute_penalty_weight(epoch)
        elbo = reconst_loss + self.penalty_scale * penalty_weight * end_penalty_loss + self.KL_beta * (kl_gp + kl_gaussian)

        return elbo, reconst_loss, kl_pi, end_penalty_loss, kl_gp, kl_gaussian, p_m, p_v, latent_samples


    def _compute_penalty_weight(self, epoch, n_epochs_warmup=400, max_weight=1., min_weight=0.):
        slope = min_weight - max_weight
        if epoch < n_epochs_warmup:
            return slope * (epoch / n_epochs_warmup) + max_weight
        return min_weight


    def encoder_inference(self, x, y_spliced, y_unspliced):
        y_spliced_ = y_spliced
        y_unspliced_ = y_unspliced

        if self.log_variational:
            y_spliced_ = torch.log(0.01 + y_spliced)
            y_unspliced_ = torch.log(0.01 + y_unspliced)

        y = torch.cat((y_spliced_, y_unspliced_), dim=-1)
        b = y.shape[0]
        qnet_mu, qnet_var = self.encoder(y)

        gp_mu = qnet_mu[:, 0:self.GP_dim]
        gp_var = qnet_var[:, 0:self.GP_dim]

        gaussian_mu = qnet_mu[:, self.GP_dim:]
        gaussian_var = qnet_var[:, self.GP_dim:]

        # KL term of Gaussian prior
        gaussian_prior_dist = Normal(torch.zeros_like(gaussian_mu), torch.ones_like(gaussian_var))
        gaussian_post_dist = Normal(gaussian_mu, torch.sqrt(gaussian_var))
        kl_gaussian = kl_divergence(gaussian_post_dist, gaussian_prior_dist).sum()


        if self.GP_dim <=0:
            p_m, p_v = gaussian_mu, gaussian_var
            latent_dist = Normal(gaussian_mu, torch.sqrt(gaussian_var))
            return {
                "kl_gp": torch.tensor(0., device=self.device),
                "kl_gaussian": kl_gaussian,
                "latent_dist": latent_dist,
                "p_m": p_m,
                "p_v": p_v
            }

        inside_elbo_recon, inside_elbo_kl = [], []
        gp_p_m, gp_p_v = [], []
        for l in range(self.GP_dim):
            gp_p_m_l, gp_p_v_l, mu_hat_l, A_hat_l = self.svgp.approximate_posterior_params(x, x,
                                                                    gp_mu[:, l], gp_var[:, l])
            inside_elbo_recon_l,  inside_elbo_kl_l = self.svgp.variational_loss(x=x, y=gp_mu[:, l],
                                                                    noise=gp_var[:, l], mu_hat=mu_hat_l,
                                                                    A_hat=A_hat_l)

            inside_elbo_recon.append(inside_elbo_recon_l)
            inside_elbo_kl.append(inside_elbo_kl_l)
            gp_p_m.append(gp_p_m_l)
            gp_p_v.append(gp_p_v_l)

        inside_elbo_recon = torch.stack(inside_elbo_recon, dim=-1)
        inside_elbo_kl = torch.stack(inside_elbo_kl, dim=-1)
        inside_elbo_recon = torch.sum(inside_elbo_recon)
        inside_elbo_kl = torch.sum(inside_elbo_kl)

        inside_elbo = inside_elbo_recon - (b / self.svgp.N_train) * inside_elbo_kl

        gp_p_m = torch.stack(gp_p_m, dim=1)
        gp_p_v = torch.stack(gp_p_v, dim=1)

        # cross entropy term
        gp_ce_term = gauss_cross_entropy(gp_p_m, gp_p_v, gp_mu, gp_var)
        gp_ce_term = torch.sum(gp_ce_term)

        # KL term of GP prior
        kl_gp = gp_ce_term - inside_elbo

        # SAMPLE
        p_m = torch.cat((gp_p_m, gaussian_mu), dim=1)
        p_v = torch.cat((gp_p_v, gaussian_var), dim=1)
        latent_dist = Normal(p_m, torch.sqrt(p_v))

        return {
            "kl_gp": kl_gp,
            "kl_gaussian": kl_gaussian,
            "latent_dist": latent_dist,
            "p_m": p_m,
            "p_v": p_v
        }


    def encoder_latent(self, x, y_spliced, y_unspliced):
        y_spliced_ = y_spliced
        y_unspliced_ = y_unspliced

        if self.log_variational:
            y_spliced_ = torch.log(0.01 + y_spliced)
            y_unspliced_ = torch.log(0.01 + y_unspliced)

        y = torch.cat((y_spliced_, y_unspliced_), dim=-1)
        qnet_mu, qnet_var = self.encoder(y)

        gp_mu = qnet_mu[:, 0:self.GP_dim]
        gp_var = qnet_var[:, 0:self.GP_dim]

        gaussian_mu = qnet_mu[:, self.GP_dim:]
        gaussian_var = qnet_var[:, self.GP_dim:]

        if self.GP_dim <= 0:
            p_m, p_v = gaussian_mu, gaussian_var
            latent_dist = Normal(p_m, torch.sqrt(p_v))
            return {
                "latent_dist": latent_dist,
                "p_m": p_m,
                "p_v": p_v
            }


        gp_p_m, gp_p_v = [], []
        for l in range(self.GP_dim):
            gp_p_m_l, gp_p_v_l, _, _ = self.svgp.approximate_posterior_params(x, x,
                                                                    gp_mu[:, l], gp_var[:, l])
            gp_p_m.append(gp_p_m_l)
            gp_p_v.append(gp_p_v_l)

        gp_p_m = torch.stack(gp_p_m, dim=1)
        gp_p_v = torch.stack(gp_p_v, dim=1)

        p_m = torch.cat((gp_p_m, gaussian_mu), dim=1)
        p_v = torch.cat((gp_p_v, gaussian_var), dim=1)
        latent_dist = Normal(p_m, torch.sqrt(p_v))

        return {
            "latent_dist": latent_dist,
            "p_m": p_m,
            "p_v": p_v
        }


    def decoder_generative(self, z, gamma, beta, alpha, alpha_1, lambda_alpha, latent_dim=None, y_spliced=None, y_unspliced=None):
        px_pi_alpha, px_rho, px_tau, alpha_rho = self.decoder(z, latent_dim=latent_dim, y_spliced=y_spliced, y_unspliced=y_unspliced)
        px_pi_alpha = torch.clamp(px_pi_alpha, 1e-10, None)
        px_pi = Dirichlet(px_pi_alpha).rsample()

        scale_unconstr = self.scale_unconstr
        if not self.use_rho:
            alpha_rho = torch.ones_like(alpha_rho, device=self.device)
        if self.use_raw:
            scale = F.softplus(torch.clamp(scale_unconstr, 0.01, None))
        else:
            scale = F.softplus(scale_unconstr)
        # scale = torch.clamp(scale, 0, 0.01)

        mixture_dist_s, mixture_dist_u, end_penalty = self.get_px(px_pi, px_rho, px_tau, scale, gamma, beta,
                                                                  alpha, alpha_1, alpha_rho, lambda_alpha)
        
        return {
            "px_pi": px_pi,
            "px_rho": px_rho,
            "px_tau": px_tau,
            "scale": scale,
            "alpha_rho": alpha_rho,
            "px_pi_alpha": px_pi_alpha,
            "mixture_dist_u": mixture_dist_u,
            "mixture_dist_s": mixture_dist_s,
            "end_penalty": end_penalty
        }


    def decoder_output(self, z, gamma, beta, alpha, alpha_1, lambda_alpha, latent_dim=None, y_spliced=None, y_unspliced=None):
        px_pi_alpha, px_rho, px_tau, alpha_rho = self.decoder(z, latent_dim=latent_dim, y_spliced=y_spliced, y_unspliced=y_unspliced)
        px_pi_alpha = torch.clamp(px_pi_alpha, 1e-10, None)
        px_pi = Dirichlet(px_pi_alpha).sample()

        scale_unconstr = self.scale_unconstr
        if not self.use_rho:
            alpha_rho = torch.ones_like(alpha_rho, device=self.device)
        if self.use_raw:
            scale = F.softplus(torch.clamp(scale_unconstr, 0.01, None))
        else:
            scale = F.softplus(scale_unconstr)
        # scale = torch.clamp(scale, 0, 0.01)

        mixture_dist_s, mixture_dist_u, end_penalty = self.get_px(px_pi, px_rho, px_tau, scale, gamma, beta,
                                                                  alpha, alpha_1, alpha_rho, lambda_alpha)
        
        return {
            "px_pi": px_pi,
            "px_rho": px_rho,
            "px_tau": px_tau,
            "alpha_rho": alpha_rho,
            "scale": scale,
            "px_pi_alpha": px_pi_alpha,
            "mixture_dist_u": mixture_dist_u,
            "mixture_dist_s": mixture_dist_s,
            "end_penalty": end_penalty
        }


    def _get_rates(self):
        # globals
        # degradation
        # gamma = torch.clamp(F.softplus(self.gamma_unconstr), 0, 100)
        gamma = torch.clamp(self.gamma_unconstr, 1e-8, 100)
        # splicing
        # beta = torch.clamp(F.softplus(self.beta_unconstr), 0, 100)
        beta = torch.clamp(self.beta_unconstr, 1e-8, 100)
        # transcription
        # alpha = torch.clamp(F.softplus(self.alpha_unconstr), 0, 100)
        alpha = torch.clamp(self.alpha_unconstr, 0, 100)
        if self.time_dep_transcription_rate:
            alpha_1 = torch.clamp(F.softplus(self.alpha_1_unconstr), 0, 100)
            lambda_alpha = torch.clamp(F.softplus(self.lambda_alpha_unconstr), 0, 100)
        else:
            alpha_1 = self.alpha_1_unconstr
            lambda_alpha = self.lambda_alpha_unconstr

        return gamma, beta, alpha, alpha_1, lambda_alpha


    def get_px(self, px_pi, px_rho, px_tau, scale, gamma, beta, alpha, alpha_1, alpha_rho, lambda_alpha):
        t_s = torch.clamp(F.softplus(self.switch_time_unconstr), 0, self.t_max)

        n_cells = px_pi.shape[0]

        # component dist
        comp_dist = Categorical(probs=px_pi)

        # compute spatial specific alpha
        alpha = alpha.reshape(1, -1) * alpha_rho
        alpha_1 = alpha_1.reshape(1, -1) * alpha_rho

        # induction
        mean_u_ind, mean_s_ind = self._get_induction_unspliced_spliced(
            alpha, alpha_1, lambda_alpha, beta, gamma, t_s * px_rho
        )

        if self.time_dep_transcription_rate:
            mean_u_ind_steady = (alpha_1 / beta).expand(n_cells, self.n_genes)
            mean_s_ind_steady = (alpha_1 / gamma).expand(n_cells, self.n_genes)
        else:
            mean_u_ind_steady = (alpha / beta).expand(n_cells, self.n_genes)
            mean_s_ind_steady = (alpha / gamma).expand(n_cells, self.n_genes)
        scale_u = scale[: self.n_genes].expand(n_cells, self.n_genes).sqrt()

        # repression
        u_0, s_0 = self._get_induction_unspliced_spliced(
            alpha, alpha_1, lambda_alpha, beta, gamma, t_s
        )

        tau = px_tau
        mean_u_rep, mean_s_rep = self._get_repression_unspliced_spliced(
            u_0,
            s_0,
            beta,
            gamma,
            (self.t_max - t_s) * tau,
        )
        mean_u_rep_steady = torch.zeros_like(mean_u_ind)
        mean_s_rep_steady = torch.zeros_like(mean_u_ind)
        scale_s = scale[self.n_genes :].expand(n_cells, self.n_genes).sqrt()

        end_penalty = ((u_0 - self.switch_unspliced).pow(2)).sum() + (
            (s_0 - self.switch_spliced).pow(2)
        ).sum()

        # unspliced
        mean_u = torch.stack(
            (mean_u_ind, mean_u_ind_steady, mean_u_rep, mean_u_rep_steady),
            dim=2,
        )
        scale_u = torch.stack(
            (
                scale_u,
                scale_u,
                scale_u,
                0.1 * scale_u,
            ),
            dim=2,
        )
        if self.use_raw:
            dist_u = NegativeBinomial(mu=mean_u, theta=scale_u)
        else:
            dist_u = Normal(mean_u, scale_u)
        mixture_dist_u = MixtureSameFamily(comp_dist, dist_u)

        # spliced
        mean_s = torch.stack(
            (mean_s_ind, mean_s_ind_steady, mean_s_rep, mean_s_rep_steady),
            dim=2,
        )
        # print("scale_s", scale_s.max(), scale_s.min())
        scale_s = torch.stack(
            (
                scale_s,
                scale_s,
                scale_s,
                0.1 * scale_s,
            ),
            dim=2,
        )
        if self.use_raw:
            dist_s = NegativeBinomial(mu=mean_s, theta=scale_s)
        else:
            dist_s = Normal(mean_s, scale_s)
        mixture_dist_s = MixtureSameFamily(comp_dist, dist_s)

        return mixture_dist_s, mixture_dist_u, end_penalty


    def _get_induction_unspliced_spliced(self, alpha, alpha_1, lambda_alpha, beta, gamma, t, eps=1e-6):
        if self.time_dep_transcription_rate:
            unspliced = alpha_1 / beta * (1 - torch.exp(-beta * t)) - (
                alpha_1 - alpha
            ) / (beta - lambda_alpha) * (
                torch.exp(-lambda_alpha * t) - torch.exp(-beta * t)
            )

            spliced = (
                alpha_1 / gamma * (1 - torch.exp(-gamma * t))
                + alpha_1
                / (gamma - beta + eps)
                * (torch.exp(-gamma * t) - torch.exp(-beta * t))
                - beta
                * (alpha_1 - alpha)
                / (beta - lambda_alpha + eps)
                / (gamma - lambda_alpha + eps)
                * (torch.exp(-lambda_alpha * t) - torch.exp(-gamma * t))
                + beta
                * (alpha_1 - alpha)
                / (beta - lambda_alpha + eps)
                / (gamma - beta + eps)
                * (torch.exp(-beta * t) - torch.exp(-gamma * t))
            )
        else:
            unspliced = (alpha / beta) * (1 - torch.exp(-beta * t))
            spliced = (alpha / gamma) * (1 - torch.exp(-gamma * t)) + (
                alpha / ((gamma - beta) + eps)
            ) * (torch.exp(-gamma * t) - torch.exp(-beta * t))

        return unspliced, spliced


    def _get_repression_unspliced_spliced(self, u_0, s_0, beta, gamma, t, eps=1e-6):
        unspliced = torch.exp(-beta * t) * u_0
        spliced = s_0 * torch.exp(-gamma * t) - (
            beta * u_0 / ((gamma - beta) + eps)
        ) * (torch.exp(-gamma * t) - torch.exp(-beta * t))
        return unspliced, spliced


    def train_model(self, adata, prior_prob, spatial_layer="spatial", spliced_layer="Ms", unspliced_layer="Mu", lr=0.01, weight_decay=0.001, 
                    batch_size=512, n_samples=1, train_size=0.95, maxiter=5000, patience=200, save_model=True, model_weights="model.pt", 
                    print_kernel_scale=True, log_dir=".", gamma_r2=None):
        """
        Model training.

        Parameters:
        -----------
        pos: array_like, shape (n_spots, 2)
            Location information.
        ncounts: array_like, shape (n_spots, n_genes)
            Preprocessed count matrix.
        raw_counts: array_like, shape (n_spots, n_genes)
            Raw count matrix.
        size_factor: array_like, shape (n_spots)
            The size factor of each spot, which need for the NB loss.
        lr: float, defalut = 0.001
            Learning rate for the opitimizer.
        weight_decay: float, default = 0.001
            Weight decay for the opitimizer.
        train_size: float, default = 0.95
            proportion of training size, the other samples are validations.
        maxiter: int, default = 5000
            Maximum number of iterations.
        patience: int, default = 200
            Patience for early stopping.
        model_weights: str
            File name to save the model weights.
        print_kernel_scale: bool
            Whether to print current kernel scale during training steps.
        """

#        self.train()

        pos = adata.obsm[spatial_layer]
        spliced = adata.layers[spliced_layer]
        unspliced = adata.layers[unspliced_layer]

        # Set data ranges for prior probability calculation

        # Create dataset with or without prior probabilities
        print("Computing velocity-based prior probabilities...")
        spliced_tensor = torch.tensor(spliced, dtype=self.dtype, device=self.device)
        unspliced_tensor = torch.tensor(unspliced, dtype=self.dtype, device=self.device)
        
        # Create dataset with prior probabilities
        dataset = TensorDataset(
            torch.tensor(pos, dtype=self.dtype), 
            torch.tensor(spliced, dtype=self.dtype), 
            torch.tensor(unspliced, dtype=self.dtype),
            torch.tensor(prior_prob, dtype=self.dtype),
        )
        gamma_r2 = torch.tensor(gamma_r2, dtype=self.dtype, device=self.device)

        if train_size < 1:
            train_dataset, validate_dataset = random_split(dataset=dataset, lengths=[train_size, 1.-train_size])
            validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        else:
            train_dataset = dataset

        if spliced.shape[0]*train_size > batch_size:
            dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        else:
            dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        early_stopping = EarlyStopping(patience=patience, modelfile=model_weights)

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, weight_decay=weight_decay)

        queue = deque()
        self.kinetic_recorder = KineticRecoder(result_dir=log_dir, gene=adata.var.index.tolist())

        print("Training")

        for epoch in range(maxiter):
            elbo_val = 0
            reconst_loss_val = 0
            end_penalty_val = 0
            kl_gp_val = 0
            kl_pi_val = 0
            kl_gaussian_val = 0
            num = 0
            self.train()
            for batch_data in enumerate(dataloader):
                batch_idx, batch_items = batch_data
                
                # Dataset includes prior probabilities
                x_batch, y_spliced_batch, y_unspliced_batch, batch_prior_prob = batch_items
                x_batch = x_batch.to(self.device)
                y_spliced_batch = y_spliced_batch.to(self.device)
                y_unspliced_batch = y_unspliced_batch.to(self.device)
                batch_prior_prob = batch_prior_prob.to(self.device)

                elbo, reconst_loss, kl_pi, end_penalty_loss, kl_gp, kl_gaussian, _, _, _ = \
                    self.forward(x=x_batch, y_spliced=y_spliced_batch, y_unspliced=y_unspliced_batch, epoch=epoch, n_samples=n_samples, prior_prob=batch_prior_prob, gamma_r2=gamma_r2)

                self.zero_grad()
                elbo.backward(retain_graph=True)
                optimizer.step()

                elbo_val += elbo.item()
                reconst_loss_val += reconst_loss.item()
                end_penalty_val += end_penalty_loss.item()
                kl_gp_val += kl_gp.item()
                kl_pi_val += kl_pi.item()
                kl_gaussian_val += kl_gaussian.item()

                num += x_batch.shape[0]

                if self.dynamicVAE:
                    KL_val = (kl_gp.item() + kl_gaussian.item()) / x_batch.shape[0]
                    queue.append(KL_val)
                    avg_KL = np.mean(queue)
                    self.KL_beta, _ = self.PID.pid(self.KL_loss*(self.GP_dim+self.Normal_dim), avg_KL)
                    if len(queue) >= 10:
                        queue.popleft()


            elbo_val = elbo_val/num
            reconst_loss_val = reconst_loss_val/num
            end_penalty_val = end_penalty_val/num
            kl_gp_val = kl_gp_val/num
            kl_pi_val = kl_pi_val/num
            kl_gaussian_val = kl_gaussian_val/num
            self.kinetic_recorder.update(self.get_rates(), epoch)
            self.kinetic_recorder.save()

            print('Training epoch {}, ELBO:{:.8f}, '
                  'Reconstuction:{:.8f}, '
                  'End penalty:{:8f}, '
                  'GP KLD:{:.8f}, '
                  'PI KLD:{:.8f}, '
                  'Gaussian KLD:{:.8f}'.format(epoch+1, elbo_val, reconst_loss_val, end_penalty_val, kl_gp_val, kl_pi_val, kl_gaussian_val))
            print('Current beta', self.KL_beta)
            if print_kernel_scale:
                print('Current kernel scale', torch.clamp(F.softplus(self.svgp.kernel.scale), min=1e-10, max=1e4).data)

            if train_size < 1:
                self.eval()
                validate_elbo_val = 0
                validate_num = 0
                for validate_batch_data in enumerate(validate_dataloader):
                    validate_batch_idx, validate_batch_items = validate_batch_data
                    
                    validate_x_batch, validate_y_spliced_batch, validate_y_unspliced_batch, validate_batch_prior_prob = validate_batch_items
                    validate_x_batch = validate_x_batch.to(self.device)
                    validate_y_spliced_batch = validate_y_spliced_batch.to(self.device)
                    validate_y_unspliced_batch = validate_y_unspliced_batch.to(self.device)
                    validate_batch_prior_prob = validate_batch_prior_prob.to(self.device)

                    validate_elbo, _, _, _, _, _, _, _, _ = \
                        self.forward(x=validate_x_batch, y_spliced=validate_y_spliced_batch, y_unspliced=validate_y_unspliced_batch, epoch=epoch, n_samples=n_samples, prior_prob=validate_batch_prior_prob, gamma_r2=gamma_r2)

                    validate_elbo_val += validate_elbo.item()
                    validate_num += validate_x_batch.shape[0]

                validate_elbo_val = validate_elbo_val / validate_num

                print('Training epoch {}, validating ELBO:{:.8f}'.format(epoch+1, validate_elbo_val))
                early_stopping(validate_elbo_val, self)
                if early_stopping.early_stop:
                    print('EarlyStopping: run {} iteration'.format(epoch+1))
                    break

        if save_model:
            torch.save(self.state_dict(), model_weights)


    def get_train_encode(self, spliced, unspliced, batch_size):
        dataset = TensorDataset(torch.tensor(spliced, dtype=self.dtype), torch.tensor(unspliced, dtype=self.dtype))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        qnet_MU, qnet_VAR = [], []
        for batch_idx, (y_spliced, y_unspliced) in enumerate(dataloader):
            y_spliced_ = y_spliced.to(self.device)
            y_unspliced_ = y_unspliced.to(self.device)

            if self.log_variational:
                y_spliced_ = torch.log(0.01 + y_spliced)
                y_unspliced_ = torch.log(0.01 + y_unspliced)

            y = torch.cat((y_spliced_, y_unspliced_), dim=-1)
            qnet_mu, qnet_var = self.encoder(y)
            qnet_MU.append(qnet_mu)
            qnet_VAR.append(qnet_var)
        qnet_MU = torch.cat(qnet_MU, dim=0)
        qnet_VAR = torch.cat(qnet_VAR, dim=0)

        return {
            "qnet_MU": qnet_MU,
            "qnet_VAR": qnet_VAR
        }


    def predict_encoder_latent(self, test_pos, train_POS, qnet_MU, qnet_VAR):
        gp_mu = qnet_MU[:, 0:self.GP_dim]
        gp_var = qnet_VAR[:, 0:self.GP_dim]

        def find_nearest(array, value):
            idx = torch.argmin(torch.sum((array - value)**2, dim=1))
            return idx

        # x_train_select_batch represents the nearest X_train spots to x_test
        x_train_select_batch = []
        for e in range(test_pos.shape[0]):
            x_train_select_batch.append(find_nearest(train_POS, test_pos[e]))
        x_train_select_batch = torch.stack(x_train_select_batch)
        gaussian_mu = qnet_MU[x_train_select_batch.long(), self.GP_dim:]
        gaussian_var = qnet_VAR[x_train_select_batch.long(), self.GP_dim:]

        gp_p_m, gp_p_v = [], []
        for l in range(self.GP_dim):
            gp_p_m_l, gp_p_v_l, _, _ = self.svgp.approximate_posterior_params(index_points_test=test_pos, index_points_train=train_POS, 
                                    y=gp_mu[:, l], noise=gp_var[:, l])
            gp_p_m.append(gp_p_m_l)
            gp_p_v.append(gp_p_v_l)

        gp_p_m = torch.stack(gp_p_m, dim=1)
        gp_p_v = torch.stack(gp_p_v, dim=1)

        # SAMPLE
        p_m = torch.cat((gp_p_m, gaussian_mu), dim=1)
        p_v = torch.cat((gp_p_v, gaussian_var), dim=1)
        latent_dist = Normal(p_m, torch.sqrt(p_v))

        return {
            "latent_dist": latent_dist,
            "p_m": p_m,
            "p_v": p_v
        }

    @torch.inference_mode()
    def get_encoder_latent(self, adata, spatial_layer="spatial", spliced_layer="Ms", unspliced_layer="Mu", 
                             indices=None, gene_list=None, hard_assignment=False, n_samples=25, batch_size=None, return_mean=True,
                             return_numpy=None):
        """Returns cells by genes by states probabilities.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        hard_assignment
            Return a hard state assignment
        n_samples
            Number of posterior samples to use for estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """

        self.eval()

        if indices is None:
            indices = np.arange(adata.n_obs)

        adata = adata[indices]

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            return_numpy = True

        pos = adata.obsm[spatial_layer]
        spliced = adata.layers[spliced_layer]
        unspliced = adata.layers[unspliced_layer]

        dataset = TensorDataset(torch.tensor(pos, dtype=self.dtype), torch.tensor(spliced, dtype=self.dtype), 
                        torch.tensor(unspliced, dtype=self.dtype))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        latent = []
        gamma, beta, alpha, alpha_1, lambda_alpha = self._get_rates()
        for batch_idx, (x_batch, y_spliced_batch, y_unspliced_batch) in enumerate(dataloader):
            x_batch = x_batch.to(self.device)
            y_spliced_batch = y_spliced_batch.to(self.device)
            y_unspliced_batch = y_unspliced_batch.to(self.device)

            latent_outputs = self.encoder_latent(x=x_batch, y_spliced=y_spliced_batch, y_unspliced=y_unspliced_batch)

            latent_dist = latent_outputs["latent_dist"]

            latent_samples = []
            for _ in range(n_samples):
                latent_samples_ = latent_dist.sample()
                latent_samples.append(latent_samples_.detach().cpu().numpy())

            latent.append(np.stack(latent_samples, axis=0))
            if return_mean:
                latent[-1] = np.mean(latent[-1], axis=0)

        latent =  np.concatenate(latent, axis=0)

        return latent

    @torch.inference_mode()
    def get_state_assignment(self, adata, spatial_layer="spatial", spliced_layer="Ms", unspliced_layer="Mu", 
                             indices=None, gene_list=None, hard_assignment=False, n_samples=25, batch_size=None, return_mean=True,
                             return_numpy=None):
        """Returns cells by genes by states probabilities.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        hard_assignment
            Return a hard state assignment
        n_samples
            Number of posterior samples to use for estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """

        self.eval()

        if indices is None:
            indices = np.arange(adata.n_obs)

        adata = adata[indices]

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            return_numpy = True

        pos = adata.obsm[spatial_layer]
        spliced = adata.layers[spliced_layer]
        unspliced = adata.layers[unspliced_layer]

        dataset = TensorDataset(torch.tensor(pos, dtype=self.dtype), torch.tensor(spliced, dtype=self.dtype), 
                        torch.tensor(unspliced, dtype=self.dtype))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        states = []
        gamma, beta, alpha, alpha_1, lambda_alpha = self._get_rates()
        for batch_idx, (x_batch, y_spliced_batch, y_unspliced_batch) in enumerate(dataloader):
            x_batch = x_batch.to(self.device)
            y_spliced_batch = y_spliced_batch.to(self.device)
            y_unspliced_batch = y_unspliced_batch.to(self.device)

            latent_outputs = self.encoder_latent(x=x_batch, y_spliced=y_spliced_batch, y_unspliced=y_unspliced_batch)

            latent_dist = latent_outputs["latent_dist"]

            latent_samples = []
            for _ in range(n_samples):
                latent_samples_ = latent_dist.sample()
                latent_samples.append(latent_samples_)

            minibatch_samples = []
            for z in latent_samples:
                generative_outputs = self.decoder_output(z, gamma, beta, alpha, alpha_1, lambda_alpha, latent_dim=None, y_spliced=y_spliced_batch, y_unspliced=y_unspliced_batch)

                output = generative_outputs["px_pi"]
                output = output[..., gene_mask, :]
                output = output.cpu().numpy()
                minibatch_samples.append(output)
            # samples by cells by genes by four
            states.append(np.stack(minibatch_samples, axis=0))
            if return_mean:
                states[-1] = np.mean(states[-1], axis=0)

        states = np.concatenate(states, axis=0)
        state_cats = [
            "induction",
            "induction_steady",
            "repression",
            "repression_steady",
        ]
        if hard_assignment and return_mean:
            hard_assign = states.argmax(-1)

            hard_assign = pd.DataFrame(
                data=hard_assign, index=adata.obs_names, columns=adata.var_names
            )
            for i, s in enumerate(state_cats):
                hard_assign = hard_assign.replace(i, s)

            states = hard_assign

        return states, state_cats

    @torch.inference_mode()
    def get_rho(self, adata, spatial_layer="spatial", spliced_layer="Ms", unspliced_layer="Mu", 
                indices=None, gene_list=None, hard_assignment=False, n_samples=25, batch_size=None, return_mean=True,
                return_numpy=None):
        """Returns cells by genes by states probabilities.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        hard_assignment
            Return a hard state assignment
        n_samples
            Number of posterior samples to use for estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """

        self.eval()

        if indices is None:
            indices = np.arange(adata.n_obs)

        adata = adata[indices]

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            return_numpy = True

        pos = adata.obsm[spatial_layer]
        spliced = adata.layers[spliced_layer]
        unspliced = adata.layers[unspliced_layer]

        dataset = TensorDataset(torch.tensor(pos, dtype=self.dtype), torch.tensor(spliced, dtype=self.dtype), 
                        torch.tensor(unspliced, dtype=self.dtype))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        rho = []
        gamma, beta, alpha, alpha_1, lambda_alpha = self._get_rates()
        for batch_idx, (x_batch, y_spliced_batch, y_unspliced_batch) in enumerate(dataloader):
            x_batch = x_batch.to(self.device)
            y_spliced_batch = y_spliced_batch.to(self.device)
            y_unspliced_batch = y_unspliced_batch.to(self.device)

            latent_outputs = self.encoder_latent(x=x_batch, y_spliced=y_spliced_batch, y_unspliced=y_unspliced_batch)

            latent_dist = latent_outputs["latent_dist"]

            latent_samples = []
            for _ in range(n_samples):
                latent_samples_ = latent_dist.sample()
                latent_samples.append(latent_samples_)

            minibatch_samples = []
            for z in latent_samples:
                generative_outputs = self.decoder_output(z, gamma, beta, alpha, alpha_1, lambda_alpha, latent_dim=None, y_spliced=y_spliced_batch, y_unspliced=y_unspliced_batch)

                output = generative_outputs["alpha_rho"]
                output = output[..., gene_mask, :]
                output = output.cpu().numpy()
                minibatch_samples.append(output)
            # samples by cells by genes by four
            rho.append(np.stack(minibatch_samples, axis=0))
            if return_mean:
                rho[-1] = np.mean(rho[-1], axis=0)

        rho = np.concatenate(rho, axis=0)

        return rho

    @torch.inference_mode()
    def predict_spatial_state_assignment(self, test_pos, adata, spatial_layer="spatial", spliced_layer="Ms", unspliced_layer="Mu", 
                                         indices=None, gene_list=None, hard_assignment=False, n_samples=1, n_samples_overall=None,
                                         batch_size=None, return_mean=True, return_numpy=None):
        """Returns cells by genes by states probabilities.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        hard_assignment
            Return a hard state assignment
        n_samples
            Number of posterior samples to use for estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """

        self.eval()

        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)

        adata = adata[indices]

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            return_numpy = True

        pos = adata.obsm[spatial_layer]
        pos = torch.tensor(pos, dtype=self.dtype).to(self.device)
        spliced = adata.layers[spliced_layer]
        unspliced = adata.layers[unspliced_layer]

        train_encode = self.get_train_encode(spliced=spliced, unspliced=unspliced, batch_size=batch_size)
        qnet_MU = train_encode["qnet_MU"]
        qnet_VAR= train_encode["qnet_VAR"]

        dataset = TensorDataset(torch.tensor(test_pos, dtype=self.dtype))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        states = []
        gamma, beta, alpha, alpha_1, lambda_alpha = self._get_rates()
        for batch_idx, (test_pos_batch) in enumerate(dataloader):
            test_pos_batch = test_pos_batch[0].to(self.device)

            latent_outputs = self.predict_encoder_latent(test_pos=test_pos_batch, train_POS=pos, qnet_MU=qnet_MU, qnet_VAR=qnet_VAR)

            latent_dist = latent_outputs["latent_dist"]

            latent_samples = []
            for _ in range(n_samples):
                latent_samples_ = latent_dist.sample()
                latent_samples.append(latent_samples_)

            minibatch_samples = []
            for z in latent_samples:
                generative_outputs = self.decoder_output(z, gamma, beta, alpha, alpha_1, lambda_alpha, latent_dim=None, y_spliced=None, y_unspliced=None)

                output = generative_outputs["px_pi"]
                output = output[..., gene_mask, :]
                output = output.cpu().numpy()
                minibatch_samples.append(output)
            # samples by cells by genes by four
            states.append(np.stack(minibatch_samples, axis=0))
            if return_mean:
                states[-1] = np.mean(states[-1], axis=0)

        states = np.concatenate(states, axis=0)
        state_cats = [
            "induction",
            "induction_steady",
            "repression",
            "repression_steady",
        ]
        if hard_assignment and return_mean:
            hard_assign = states.argmax(-1)

            hard_assign = pd.DataFrame(
                data=hard_assign, index=adata.obs_names, columns=adata.var_names
            )
            for i, s in enumerate(state_cats):
                hard_assign = hard_assign.replace(i, s)

            states = hard_assign

        return states, state_cats


    @torch.inference_mode()
    def get_latent_time(self, adata, spatial_layer="spatial", spliced_layer="Ms", unspliced_layer="Mu", 
                        indices=None, gene_list=None, time_statistic="mean", n_samples=1, n_samples_overall=None,
                        batch_size=None, return_mean=True, return_numpy=None):
        """
        Returns the cells by genes latent time.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        time_statistic
            Whether to compute expected time over states, or maximum a posteriori time over maximal
            probability state.
        n_samples
            Number of posterior samples to use for estimation.
        n_samples_overall
            Number of overall samples to return. Setting this forces n_samples=1.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """

        self.eval()

        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)

        adata = adata[indices]

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            return_numpy = True

        pos = adata.obsm[spatial_layer]
        spliced = adata.layers[spliced_layer]
        unspliced = adata.layers[unspliced_layer]

        dataset = TensorDataset(torch.tensor(pos, dtype=self.dtype), torch.tensor(spliced, dtype=self.dtype), 
                        torch.tensor(unspliced, dtype=self.dtype))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        times = []
        gamma, beta, alpha, alpha_1, lambda_alpha = self._get_rates()
        for batch_idx, (x_batch, y_spliced_batch, y_unspliced_batch) in enumerate(dataloader):
            x_batch = x_batch.to(self.device)
            y_spliced_batch = y_spliced_batch.to(self.device)
            y_unspliced_batch = y_unspliced_batch.to(self.device)

            latent_outputs = self.encoder_latent(x=x_batch, y_spliced=y_spliced_batch, y_unspliced=y_unspliced_batch)

            latent_dist = latent_outputs["latent_dist"]

            latent_samples = []
            for _ in range(n_samples):
                latent_samples_ = latent_dist.sample()
                latent_samples.append(latent_samples_)

            minibatch_samples = []
            for z in latent_samples:
                generative_outputs = self.decoder_output(z, gamma, beta, alpha, alpha_1, lambda_alpha, latent_dim=None, y_spliced=y_spliced_batch, y_unspliced=y_unspliced_batch)

                pi = generative_outputs["px_pi"]
                ind_prob = pi[..., 0]
                steady_prob = pi[..., 1]
                rep_prob = pi[..., 2]
                rep_steady_prob = pi[..., 3]
                switch_time = torch.clamp(F.softplus(self.switch_time_unconstr), 0, self.t_max)

                ind_time = generative_outputs["px_rho"] * switch_time
                rep_time = switch_time + (
                    generative_outputs["px_tau"] * (self.t_max - switch_time)
                )

                if time_statistic == "mean":
                    output = (
                        ind_prob * ind_time
                        + rep_prob * rep_time
                        + steady_prob * switch_time
                        # + rep_steady_prob * self.t_max
                    )
                else:
                    t = torch.stack(
                        [
                            ind_time,
                            switch_time.expand(ind_time.shape),
                            rep_time,
                            torch.zeros_like(ind_time),
                        ],
                        dim=2,
                    )
                    max_prob = torch.amax(pi, dim=-1)
                    max_prob = torch.stack([max_prob] * 4, dim=2)
                    max_prob_mask = pi.ge(max_prob)
                    output = (t * max_prob_mask).sum(dim=-1)

                output = output[..., gene_mask]
                output = output.cpu().numpy()
                minibatch_samples.append(output)
            # samples by cells by genes by four
            times.append(np.stack(minibatch_samples, axis=0))
            if return_mean:
                times[-1] = np.mean(times[-1], axis=0)

        if n_samples > 1:
            # The -2 axis correspond to cells.
            times = np.concatenate(times, axis=-2)
        else:
            times = np.concatenate(times, axis=0)

        if return_numpy is None or return_numpy is False:
            return pd.DataFrame(
                times,
                columns=adata.var_names[gene_mask],
                index=adata.obs_names[indices],
            )
        else:
            return times


    @torch.inference_mode()
    def predict_spatial_latent_time(self, test_pos, adata, spatial_layer="spatial", spliced_layer="Ms", unspliced_layer="Mu", 
                                    indices=None, gene_list=None, time_statistic="mean", n_samples=1, n_samples_overall=None,
                                    batch_size=None, return_mean=True, return_numpy=None):
        """
        Returns the cells by genes latent time.

        Parameters
        ----------
        test_pos
            Spatial position of testing set.
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        time_statistic
            Whether to compute expected time over states, or maximum a posteriori time over maximal
            probability state.
        n_samples
            Number of posterior samples to use for estimation.
        n_samples_overall
            Number of overall samples to return. Setting this forces n_samples=1.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """

        self.eval()

        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)

        adata = adata[indices]

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            return_numpy = True

        pos = adata.obsm[spatial_layer]
        pos = torch.tensor(pos, dtype=self.dtype).to(self.device)
        spliced = adata.layers[spliced_layer]
        unspliced = adata.layers[unspliced_layer]

        train_encode = self.get_train_encode(spliced=spliced, unspliced=unspliced, batch_size=batch_size)
        qnet_MU = train_encode["qnet_MU"]
        qnet_VAR= train_encode["qnet_VAR"]

        dataset = TensorDataset(torch.tensor(test_pos, dtype=self.dtype))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        times = []
        gamma, beta, alpha, alpha_1, lambda_alpha = self._get_rates()
        for batch_idx, (test_pos_batch) in enumerate(dataloader):
            test_pos_batch = test_pos_batch[0].to(self.device)

            latent_outputs = self.predict_encoder_latent(test_pos=test_pos_batch, train_POS=pos, qnet_MU=qnet_MU, qnet_VAR=qnet_VAR)

            latent_dist = latent_outputs["latent_dist"]

            latent_samples = []
            for _ in range(n_samples):
                latent_samples_ = latent_dist.sample()
                latent_samples.append(latent_samples_)

            minibatch_samples = []
            for z in latent_samples:
                generative_outputs = self.decoder_output(z, gamma, beta, alpha, alpha_1, lambda_alpha, latent_dim=None, y_spliced=None, y_unspliced=None)

                pi = generative_outputs["px_pi"]
                ind_prob = pi[..., 0]
                steady_prob = pi[..., 1]
                rep_prob = pi[..., 2]
                # rep_steady_prob = pi[..., 3]
                switch_time = torch.clamp(F.softplus(self.switch_time_unconstr), 0, self.t_max)

                ind_time = generative_outputs["px_rho"] * switch_time
                rep_time = switch_time + (
                    generative_outputs["px_tau"] * (self.t_max - switch_time)
                )

                if time_statistic == "mean":
                    output = (
                        ind_prob * ind_time
                        + rep_prob * rep_time
                        + steady_prob * switch_time
                        # + rep_steady_prob * self.t_max
                    )
                else:
                    t = torch.stack(
                        [
                            ind_time,
                            switch_time.expand(ind_time.shape),
                            rep_time,
                            torch.zeros_like(ind_time),
                        ],
                        dim=2,
                    )
                    max_prob = torch.amax(pi, dim=-1)
                    max_prob = torch.stack([max_prob] * 4, dim=2)
                    max_prob_mask = pi.ge(max_prob)
                    output = (t * max_prob_mask).sum(dim=-1)

                output = output[..., gene_mask]
                output = output.cpu().numpy()
                minibatch_samples.append(output)
            # samples by cells by genes by four
            times.append(np.stack(minibatch_samples, axis=0))
            if return_mean:
                times[-1] = np.mean(times[-1], axis=0)

        if n_samples > 1:
            # The -2 axis correspond to cells.
            times = np.concatenate(times, axis=-2)
        else:
            times = np.concatenate(times, axis=0)

        if return_numpy is None or return_numpy is False:
            return pd.DataFrame(
                times,
                columns=adata.var_names[gene_mask],
            )
        else:
            return times


    @torch.inference_mode()
    def get_velocity(self, adata, spatial_layer="spatial", spliced_layer="Ms", unspliced_layer="Mu", 
                     indices=None, gene_list=None, n_samples=1, n_samples_overall=None, batch_size=None, 
                     return_mean=True, return_numpy=None, velo_statistic="mean", velo_mode="spliced", clip=True):
        """Returns cells by genes velocity estimates.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        gene_list
            Return velocities for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        n_samples
            Number of posterior samples to use for estimation for each cell.
        n_samples_overall
            Number of overall samples to return. Setting this forces n_samples=1.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.
        velo_statistic
            Whether to compute expected velocity over states, or maximum a posteriori velocity over maximal
            probability state.
        velo_mode
            Compute ds/dt or du/dt.
        clip
            Clip to minus spliced value

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """

        self.eval()

        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)

        adata = adata[indices]

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            return_numpy = True

        pos = adata.obsm[spatial_layer]
        spliced = adata.layers[spliced_layer]
        unspliced = adata.layers[unspliced_layer]

        dataset = TensorDataset(torch.tensor(pos, dtype=self.dtype), torch.tensor(spliced, dtype=self.dtype), 
                        torch.tensor(unspliced, dtype=self.dtype))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        velos = []
        gamma, beta, alpha, alpha_1, lambda_alpha = self._get_rates()
        for batch_idx, (x_batch, y_spliced_batch, y_unspliced_batch) in enumerate(dataloader):
            x_batch = x_batch.to(self.device)
            y_spliced_batch = y_spliced_batch.to(self.device)
            y_unspliced_batch = y_unspliced_batch.to(self.device)

            latent_outputs = self.encoder_latent(x=x_batch, y_spliced=y_spliced_batch, y_unspliced=y_unspliced_batch)

            latent_dist = latent_outputs["latent_dist"]

            latent_samples = []
            for _ in range(n_samples):
                latent_samples_ = latent_dist.sample()
                latent_samples.append(latent_samples_)

            minibatch_samples = []
            for z in latent_samples:
                generative_outputs = self.decoder_output(z, gamma, beta, alpha, alpha_1, lambda_alpha, latent_dim=None, y_spliced=y_spliced_batch, y_unspliced=y_unspliced_batch)

                pi = generative_outputs["px_pi"]
                tau = generative_outputs["px_tau"]
                rho = generative_outputs["px_rho"]
                alpha_rho = generative_outputs["alpha_rho"]

                ind_prob = pi[..., 0]
                steady_prob = pi[..., 1]
                rep_prob = pi[..., 2]
                rep_steady_prob = pi[..., 3]
                switch_time = torch.clamp(F.softplus(self.switch_time_unconstr), 0, self.t_max)

                curr_alpha = alpha * alpha_rho
                curr_alpha_1 = alpha_1 * alpha_rho
                ind_time = switch_time * rho
                u_0, s_0 = self._get_induction_unspliced_spliced(
                    curr_alpha, curr_alpha_1, lambda_alpha, beta, gamma, switch_time
                )
                rep_time = (self.t_max - switch_time) * tau
                mean_u_rep, mean_s_rep = self._get_repression_unspliced_spliced(
                    u_0,
                    s_0,
                    beta,
                    gamma,
                    rep_time,
                )
                if velo_mode == "spliced":
                    velo_rep = beta * mean_u_rep - gamma * mean_s_rep
                else:
                    velo_rep = -beta * mean_u_rep
                mean_u_ind, mean_s_ind = self._get_induction_unspliced_spliced(
                    curr_alpha, curr_alpha_1, lambda_alpha, beta, gamma, ind_time
                )
                if velo_mode == "spliced":
                    velo_ind = beta * mean_u_ind - gamma * mean_s_ind
                else:
                    transcription_rate = curr_alpha_1 - (curr_alpha_1 - curr_alpha) * torch.exp(
                        -lambda_alpha * ind_time
                    )
                    velo_ind = transcription_rate - beta * mean_u_ind

                if velo_mode == "spliced":
                    # velo_steady = beta * u_0 - gamma * s_0
                    velo_steady = torch.zeros_like(velo_ind)
                else:
                    # velo_steady = alpha - beta * u_0
                    velo_steady = torch.zeros_like(velo_ind)

                # expectation
                if velo_statistic == "mean":
                    output = (
                        ind_prob * velo_ind
                        + rep_prob * velo_rep
                        + steady_prob * velo_steady
                        + rep_steady_prob * torch.zeros_like(velo_rep)
                    )
                # maximum
                else:
                    v = torch.stack(
                        [
                            velo_ind,
                            velo_steady.expand(velo_ind.shape),
                            velo_rep,
                            torch.zeros_like(velo_rep),
                        ],
                        dim=2,
                    )
                    max_prob = torch.amax(pi, dim=-1)
                    max_prob = torch.stack([max_prob] * 4, dim=2)
                    max_prob_mask = pi.ge(max_prob)
                    output = (v * max_prob_mask).sum(dim=-1)

                output = output[..., gene_mask]
                output = output.cpu().numpy()
                minibatch_samples.append(output)
            # samples by cells by genes
            velos.append(np.stack(minibatch_samples, axis=0))
            if return_mean:
                # mean over samples axis
                velos[-1] = np.mean(velos[-1], axis=0)

        if n_samples > 1:
            # The -2 axis correspond to cells.
            velos = np.concatenate(velos, axis=-2)
        else:
            velos = np.concatenate(velos, axis=0)

#        spliced = self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)

        if clip:
            velos = np.clip(velos, -spliced[indices], None)

        if return_numpy is None or return_numpy is False:
            return pd.DataFrame(
                velos,
                columns=adata.var_names[gene_mask],
                index=adata.obs_names[indices],
            )
        else:
            return velos


    @torch.inference_mode()
    def predict_spatial_velocity(self, test_pos, adata, spatial_layer="spatial", spliced_layer="Ms", unspliced_layer="Mu", 
                                 indices=None, gene_list=None, n_samples=1, n_samples_overall=None, batch_size=None, 
                                 return_mean=True, return_numpy=None, velo_statistic="mean", velo_mode="spliced", clip=False):
        """Returns cells by genes velocity estimates.

        Parameters
        ----------
        test_pos
            Spatial position of testing set.
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        gene_list
            Return velocities for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        n_samples
            Number of posterior samples to use for estimation for each cell.
        n_samples_overall
            Number of overall samples to return. Setting this forces n_samples=1.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.
        velo_statistic
            Whether to compute expected velocity over states, or maximum a posteriori velocity over maximal
            probability state.
        velo_mode
            Compute ds/dt or du/dt.
        clip
            Clip to minus spliced value

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """

        self.eval()

        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)

        adata = adata[indices]

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            return_numpy = True

        pos = adata.obsm[spatial_layer]
        pos = torch.tensor(pos, dtype=self.dtype).to(self.device)
        spliced = adata.layers[spliced_layer]
        unspliced = adata.layers[unspliced_layer]

        train_encode = self.get_train_encode(spliced=spliced, unspliced=unspliced, batch_size=batch_size)
        qnet_MU = train_encode["qnet_MU"]
        qnet_VAR= train_encode["qnet_VAR"]

        dataset = TensorDataset(torch.tensor(test_pos, dtype=self.dtype))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        velos = []
        gamma, beta, alpha, alpha_1, lambda_alpha = self._get_rates()
        for batch_idx, (test_pos_batch) in enumerate(dataloader):
            test_pos_batch = test_pos_batch[0].to(self.device)

            latent_outputs = self.predict_encoder_latent(test_pos=test_pos_batch, train_POS=pos, qnet_MU=qnet_MU, qnet_VAR=qnet_VAR)

            latent_dist = latent_outputs["latent_dist"]

            latent_samples = []
            for _ in range(n_samples):
                latent_samples_ = latent_dist.sample()
                latent_samples.append(latent_samples_)

            minibatch_samples = []
            for z in latent_samples:
                generative_outputs = self.decoder_output(z, gamma, beta, alpha, alpha_1, lambda_alpha, latent_dim=None, y_spliced=None, y_unspliced=None)

                pi = generative_outputs["px_pi"]
                tau = generative_outputs["px_tau"]
                rho = generative_outputs["px_rho"]
                alpha_rho = generative_outputs["alpha_rho"]

                ind_prob = pi[..., 0]
                steady_prob = pi[..., 1]
                rep_prob = pi[..., 2]
                switch_time = torch.clamp(F.softplus(self.switch_time_unconstr), 0, self.t_max)

                curr_alpha = alpha * alpha_rho
                curr_alpha_1 = alpha_1 * alpha_rho
                ind_time = switch_time * rho
                u_0, s_0 = self._get_induction_unspliced_spliced(
                    curr_alpha, curr_alpha_1, lambda_alpha, beta, gamma, switch_time
                )
                rep_time = (self.t_max - switch_time) * tau
                mean_u_rep, mean_s_rep = self._get_repression_unspliced_spliced(
                    u_0,
                    s_0,
                    beta,
                    gamma,
                    rep_time,
                )
                if velo_mode == "spliced":
                    velo_rep = beta * mean_u_rep - gamma * mean_s_rep
                else:
                    velo_rep = -beta * mean_u_rep
                mean_u_ind, mean_s_ind = self._get_induction_unspliced_spliced(
                    curr_alpha, curr_alpha_1, lambda_alpha, beta, gamma, ind_time
                )
                if velo_mode == "spliced":
                    velo_ind = beta * mean_u_ind - gamma * mean_s_ind
                else:
                    transcription_rate = curr_alpha_1 - (curr_alpha_1 - curr_alpha) * torch.exp(
                        -lambda_alpha * ind_time
                    )
                    velo_ind = transcription_rate - beta * mean_u_ind

                if velo_mode == "spliced":
                    # velo_steady = beta * u_0 - gamma * s_0
                    velo_steady = torch.zeros_like(velo_ind)
                else:
                    # velo_steady = alpha - beta * u_0
                    velo_steady = torch.zeros_like(velo_ind)

                # expectation
                if velo_statistic == "mean":
                    output = (
                        ind_prob * velo_ind
                        + rep_prob * velo_rep
                        + steady_prob * velo_steady
                    )
                # maximum
                else:
                    v = torch.stack(
                        [
                            velo_ind,
                            velo_steady.expand(velo_ind.shape),
                            velo_rep,
                            torch.zeros_like(velo_rep),
                        ],
                        dim=2,
                    )
                    max_prob = torch.amax(pi, dim=-1)
                    max_prob = torch.stack([max_prob] * 4, dim=2)
                    max_prob_mask = pi.ge(max_prob)
                    output = (v * max_prob_mask).sum(dim=-1)

                output = output[..., gene_mask]
                output = output.cpu().numpy()
                minibatch_samples.append(output)
            # samples by cells by genes
            velos.append(np.stack(minibatch_samples, axis=0))
            if return_mean:
                # mean over samples axis
                velos[-1] = np.mean(velos[-1], axis=0)

        if n_samples > 1:
            # The -2 axis correspond to cells.
            velos = np.concatenate(velos, axis=-2)
        else:
            velos = np.concatenate(velos, axis=0)

#        spliced = self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)

        if clip:
            velos = np.clip(velos, -spliced[indices], None)

        if return_numpy is None or return_numpy is False:
            return pd.DataFrame(
                velos,
                columns=adata.var_names[gene_mask],
            )
        else:
            return velos


    @torch.inference_mode()
    def get_expression_fit(self, adata, spatial_layer="spatial", spliced_layer="Ms", unspliced_layer="Mu", 
                           indices=None, gene_list=None, n_samples=1, batch_size=None, return_mean=True, return_numpy=None,
                           restrict_to_latent_dim=None):
        """Returns the fitted spliced and unspliced abundance (s(t) and u(t)).

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        n_samples
            Number of posterior samples to use for estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """

        self.eval()

        if indices is None:
            indices = np.arange(adata.n_obs)

        adata = adata[indices]

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            return_numpy = True

        pos = adata.obsm[spatial_layer]
        spliced = adata.layers[spliced_layer]
        unspliced = adata.layers[unspliced_layer]

        dataset = TensorDataset(torch.tensor(pos, dtype=self.dtype), torch.tensor(spliced, dtype=self.dtype), 
                        torch.tensor(unspliced, dtype=self.dtype))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        fits_s = []
        fits_u = []
        gamma, beta, alpha, alpha_1, lambda_alpha = self._get_rates()
        for batch_idx, (x_batch, y_spliced_batch, y_unspliced_batch) in enumerate(dataloader):
            x_batch = x_batch.to(self.device)
            y_spliced_batch = y_spliced_batch.to(self.device)
            y_unspliced_batch = y_unspliced_batch.to(self.device)

            latent_outputs = self.encoder_latent(x=x_batch, y_spliced=y_spliced_batch, y_unspliced=y_unspliced_batch)

            latent_dist = latent_outputs["latent_dist"]

            latent_samples = []
            for _ in range(n_samples):
                latent_samples_ = latent_dist.sample()
                latent_samples.append(latent_samples_)

            minibatch_samples_s = []
            minibatch_samples_u = []
            for z in latent_samples:
                generative_outputs = self.decoder_output(z, gamma, beta, alpha, alpha_1, lambda_alpha, latent_dim=restrict_to_latent_dim, y_spliced=y_spliced_batch, y_unspliced=y_unspliced_batch)

                mixture_dist_s = generative_outputs["mixture_dist_s"]
                mixture_dist_u = generative_outputs["mixture_dist_u"]

                fit_s = mixture_dist_s.mean
                fit_u = mixture_dist_u.mean

                fit_s = fit_s[..., gene_mask]
                fit_s = fit_s.cpu().numpy()
                fit_u = fit_u[..., gene_mask]
                fit_u = fit_u.cpu().numpy()

                minibatch_samples_s.append(fit_s)
                minibatch_samples_u.append(fit_u)

            # samples by cells by genes
            fits_s.append(np.stack(minibatch_samples_s, axis=0))
            if return_mean:
                # mean over samples axis
                fits_s[-1] = np.mean(fits_s[-1], axis=0)
            # samples by cells by genes
            fits_u.append(np.stack(minibatch_samples_u, axis=0))
            if return_mean:
                # mean over samples axis
                fits_u[-1] = np.mean(fits_u[-1], axis=0)

        if n_samples > 1:
            # The -2 axis correspond to cells.
            fits_s = np.concatenate(fits_s, axis=-2)
            fits_u = np.concatenate(fits_u, axis=-2)
        else:
            fits_s = np.concatenate(fits_s, axis=0)
            fits_u = np.concatenate(fits_u, axis=0)

        if return_numpy is None or return_numpy is False:
            df_s = pd.DataFrame(
                fits_s,
                columns=adata.var_names[gene_mask],
                index=adata.obs_names[indices],
            )
            df_u = pd.DataFrame(
                fits_u,
                columns=adata.var_names[gene_mask],
                index=adata.obs_names[indices],
            )
            return df_s, df_u
        else:
            return fits_s, fits_u


    @torch.inference_mode()
    def predict_spatial_expression_fit(self, test_pos, adata, spatial_layer="spatial", spliced_layer="Ms", unspliced_layer="Mu", 
                                       indices=None, gene_list=None, n_samples=1, batch_size=None, return_mean=True, return_numpy=None,
                                       restrict_to_latent_dim=None):
        """Returns the fitted spliced and unspliced abundance (s(t) and u(t)).

        Parameters
        ----------
        test_pos
            Spatial position of testing set.
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        n_samples
            Number of posterior samples to use for estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """

        self.eval()

        if indices is None:
            indices = np.arange(adata.n_obs)

        adata = adata[indices]

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            return_numpy = True

        pos = adata.obsm[spatial_layer]
        pos = torch.tensor(pos, dtype=self.dtype).to(self.device)
        spliced = adata.layers[spliced_layer]
        unspliced = adata.layers[unspliced_layer]

        train_encode = self.get_train_encode(spliced=spliced, unspliced=unspliced, batch_size=batch_size)
        qnet_MU = train_encode["qnet_MU"]
        qnet_VAR= train_encode["qnet_VAR"]

        dataset = TensorDataset(torch.tensor(test_pos, dtype=self.dtype))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        fits_s = []
        fits_u = []
        gamma, beta, alpha, alpha_1, lambda_alpha = self._get_rates()
        for batch_idx, (test_pos_batch) in enumerate(dataloader):
            test_pos_batch = test_pos_batch[0].to(self.device)

            latent_outputs = self.predict_encoder_latent(test_pos=test_pos_batch, train_POS=pos, qnet_MU=qnet_MU, qnet_VAR=qnet_VAR)

            latent_dist = latent_outputs["latent_dist"]

            latent_samples = []
            for _ in range(n_samples):
                latent_samples_ = latent_dist.sample()
                latent_samples.append(latent_samples_)

            minibatch_samples_s = []
            minibatch_samples_u = []
            for z in latent_samples:
                generative_outputs = self.decoder_output(z, gamma, beta, alpha, alpha_1, lambda_alpha, latent_dim=restrict_to_latent_dim, y_spliced=None, y_unspliced=None)

                mixture_dist_s = generative_outputs["mixture_dist_s"]
                mixture_dist_u = generative_outputs["mixture_dist_u"]

                fit_s = mixture_dist_s.mean
                fit_u = mixture_dist_u.mean

                fit_s = fit_s[..., gene_mask]
                fit_s = fit_s.cpu().numpy()
                fit_u = fit_u[..., gene_mask]
                fit_u = fit_u.cpu().numpy()

                minibatch_samples_s.append(fit_s)
                minibatch_samples_u.append(fit_u)

            # samples by cells by genes
            fits_s.append(np.stack(minibatch_samples_s, axis=0))
            if return_mean:
                # mean over samples axis
                fits_s[-1] = np.mean(fits_s[-1], axis=0)
            # samples by cells by genes
            fits_u.append(np.stack(minibatch_samples_u, axis=0))
            if return_mean:
                # mean over samples axis
                fits_u[-1] = np.mean(fits_u[-1], axis=0)

        if n_samples > 1:
            # The -2 axis correspond to cells.
            fits_s = np.concatenate(fits_s, axis=-2)
            fits_u = np.concatenate(fits_u, axis=-2)
        else:
            fits_s = np.concatenate(fits_s, axis=0)
            fits_u = np.concatenate(fits_u, axis=0)

        if return_numpy is None or return_numpy is False:
            df_s = pd.DataFrame(
                fits_s,
                columns=adata.var_names[gene_mask],
                index=adata.obs_names[indices],
            )
            df_u = pd.DataFrame(
                fits_u,
                columns=adata.var_names[gene_mask],
                index=adata.obs_names[indices],
            )
            return df_s, df_u
        else:
            return fits_s, fits_u


    @torch.inference_mode()
    def get_gene_likelihood(self, adata, spatial_layer="spatial", spliced_layer="Ms", unspliced_layer="Mu", 
                            indices=None, gene_list=None, n_samples=1, batch_size=None, return_mean=True, return_numpy=None):
        """Returns the likelihood per gene. Higher is better.

        This is denoted as :math:`\rho_n` in the scVI paper.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        transform_batch
            Batch to condition on.
            If transform_batch is:

            - None, then real observed batch is used.
            - int, then batch transform_batch is used.
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        library_size
            Scale the expression frequencies to a common library size.
            This allows gene expression levels to be interpreted on a common scale of relevant
            magnitude. If set to `"latent"`, use the latent libary size.
        n_samples
            Number of posterior samples to use for estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """

        self.eval()

        if indices is None:
            indices = np.arange(adata.n_obs)

        adata = adata[indices]

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            return_numpy = True

        pos = adata.obsm[spatial_layer]
        spliced = adata.layers[spliced_layer]
        unspliced = adata.layers[unspliced_layer]

        dataset = TensorDataset(torch.tensor(pos, dtype=self.dtype), torch.tensor(spliced, dtype=self.dtype), 
                        torch.tensor(unspliced, dtype=self.dtype))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        rls = []
        gamma, beta, alpha, alpha_1, lambda_alpha = self._get_rates()
        for batch_idx, (x_batch, y_spliced_batch, y_unspliced_batch) in enumerate(dataloader):
            x_batch = x_batch.to(self.device)
            y_spliced_batch = y_spliced_batch.to(self.device)
            y_unspliced_batch = y_unspliced_batch.to(self.device)

            latent_outputs = self.encoder_latent(x=x_batch, y_spliced=y_spliced_batch, y_unspliced=y_unspliced_batch)

            latent_dist = latent_outputs["latent_dist"]

            latent_samples = []
            for _ in range(n_samples):
                latent_samples_ = latent_dist.sample()
                latent_samples.append(latent_samples_)

            minibatch_samples = []
            for z in latent_samples:
                generative_outputs = self.decoder_output(z, gamma, beta, alpha, alpha_1, lambda_alpha, latent_dim=None, y_spliced=y_spliced_batch, y_unspliced=y_unspliced_batch)

                mixture_dist_s = generative_outputs["mixture_dist_s"]
                mixture_dist_u = generative_outputs["mixture_dist_u"]

                reconst_loss_s = -mixture_dist_s.log_prob(y_spliced_batch.to(self.device))
                reconst_loss_u = -mixture_dist_u.log_prob(y_unspliced_batch.to(self.device))
                output = -(reconst_loss_s + reconst_loss_u)
                output = output[..., gene_mask]
                output = output.cpu().numpy()
                minibatch_samples.append(output)
            # samples by cells by genes by four
            rls.append(np.stack(minibatch_samples, axis=0))
            if return_mean:
                rls[-1] = np.mean(rls[-1], axis=0)

        rls = np.concatenate(rls, axis=0)
        return rls


    @torch.inference_mode()
    def get_rates(self):

        self.eval()

        gamma, beta, alpha, alpha_1, lambda_alpha = self._get_rates()

        return {
            "beta": beta.cpu().numpy(),
            "gamma": gamma.cpu().numpy(),
            "alpha": alpha.cpu().numpy(),
            "alpha_1": alpha_1.cpu().numpy(),
            "lambda_alpha": lambda_alpha.cpu().numpy()
        }

    @torch.inference_mode()
    def get_cell_latent_time(self, adata, group_key="leiden", spatial_layer="spatial", spliced_layer="Ms", unspliced_layer="Mu",
                             vkey="pred_velocity", leiden_resolution=0.1, root_cluster=None, 
                             min_likelihood=None, min_confidence=0.75, batch_size=None, n_samples=25):
        """计算细胞级别的latent time，基于leiden聚类和PAGA图。
        
        Parameters
        ----------
        adata : AnnData
            包含基因级别时间信息的AnnData对象
        spatial_layer : str
            空间坐标的key
        spliced_layer : str
            spliced counts的layer
        unspliced_layer : str  
            unspliced counts的layer
        vkey : str
            velocity的key
        leiden_resolution : float
            leiden聚类的分辨率参数
        root_cluster : int or None
            指定的root cluster编号，如果为None则自动选择
        min_likelihood : float or None
            最小likelihood阈值
        min_confidence : float
            最小confidence阈值
        batch_size : int
            batch size
        n_samples : int
            采样数量
            
        Returns
        -------
        cell_latent_time : np.ndarray
            细胞级别的latent time
        """

        self.eval()

        # 确保adata有必要的信息
        if "latent_z" not in adata.obsm.keys():
            print("Computing latent embedding...")
            latent_z = self.get_encoder_latent(adata, spatial_layer=spatial_layer, 
                                               spliced_layer=spliced_layer, 
                                               unspliced_layer=unspliced_layer,
                                               batch_size=batch_size, n_samples=n_samples)
            adata.obsm["latent_z"] = latent_z

        if "pred_t" not in adata.layers.keys():
            print("Computing gene-level latent time...")
            pred_t = self.get_latent_time(adata, spatial_layer=spatial_layer,
                                          spliced_layer=spliced_layer,
                                          unspliced_layer=unspliced_layer,
                                          batch_size=batch_size, n_samples=n_samples)
            adata.layers["pred_t"] = pred_t
            adata.layers["fit_t"] = pred_t

        # Step 1: 使用latent_z进行leiden聚类
        if group_key in {"leiden"}:
            print("Performing leiden clustering on latent space...")
            latent_adata = sc.AnnData(X=adata.obsm["latent_z"], 
                                    obs=adata.obs.copy(),
                                    obsm={"X_coord": adata.obsm[spatial_layer]})
            sc.pp.neighbors(latent_adata)
            sc.tl.leiden(latent_adata, resolution=leiden_resolution)
            adata.obs["leiden"] = latent_adata.obs["leiden"]
        print(f"Using {group_key} for pseudotime estimation...")

        # Step 2: 使用velocity构建PAGA图
        print("Building PAGA graph...")
        scv.pp.neighbors(adata, use_rep=spatial_layer)
        scv.tl.velocity_graph(adata, vkey=vkey)

        scv.tl.paga(adata, groups=group_key, vkey=vkey)

        # Step 3: 选择root cluster
        if root_cluster is None:
            # 自动选择：找到PAGA树中的起始节点
            paga_tree = adata.uns["paga"]["transitions_confidence"].toarray().T
            # 计算每个cluster的入度（有多少cluster指向它）
            in_degree = paga_tree.sum(axis=0)
            # 选择入度最小的cluster作为root
            root_cluster = int(np.argmin(in_degree))
            root_cluster = adata.obs[group_key].cat.categories.tolist()[root_cluster]
            print(f"Auto-selected root cluster: {root_cluster}")
        else:
            print(f"Using specified root cluster: {root_cluster}")

        # Step 4: 获取root cluster中的所有细胞
        root_cells = np.where(adata.obs[group_key] == str(root_cluster))[0].tolist()
        adata.uns["root_cells_cluster"] = root_cells
        print(f"Number of root cells: {len(root_cells)}")

        # Step 5: 计算cell-level latent time
        print("Computing cell-level latent time...")
        t = np.array(adata.layers["fit_t"])
        idx_valid = ~np.isnan(t.sum(0))

        if min_likelihood is not None and "fit_likelihood" in adata.var.keys():
            likelihood = adata.var["fit_likelihood"].values
            idx_valid &= np.array(likelihood >= min_likelihood, dtype=bool)

        t = t[:, idx_valid]
        t_sum = np.sum(t, 1)
        conn = get_connectivities(adata)

        # 使用root cells计算latent time
        roots = root_cells[:min(len(root_cells), 10)]  # 最多使用10个root cells
        latent_time = np.ones(shape=(len(roots), adata.n_obs))

        for i, root in enumerate(roots):
            t_root, t_ = root_time(t, root=root)
            latent_time[i] = compute_shared_time(t_root)

        # 保存所有root的latent time
        adata.uns["latent_time_per_root"] = latent_time
        
        # 平均所有root的结果
        latent_time_mean = scale(np.mean(latent_time, axis=0))

        # 使用confidence进行平滑
        tl = latent_time_mean
        tc = conn.dot(latent_time_mean)

        z = tl.dot(tc) / (tc.dot(tc) + 1e-10)
        tl_conf = (1 - np.abs(tl / (np.max(tl) + 1e-10) - tc * z / (np.max(tl) + 1e-10))) ** 2
        idx_low_confidence = tl_conf < min_confidence

        # 对低confidence的细胞使用邻居的平均时间
        conn_new = conn.copy()
        conn_new[:, idx_low_confidence] = 0
        conn_new.eliminate_zeros()
        latent_time_smoothed = conn_new.dot(latent_time_mean)
        
        # 最终scale
        cell_latent_time = scale(latent_time_smoothed)
        
        # 保存到adata
        adata.obs["cell_latent_time"] = cell_latent_time
        
        logg.info("    finished", r=True)
        logg.hint("added \n" "    'cell_latent_time', cell-level latent time (adata.obs)")
        
        return cell_latent_time
