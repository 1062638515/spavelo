from spaVelo import SPAVELO
import torch
import torch.nn.functional as F
from torch.distributions import kl_divergence, Dirichlet
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd


class SPAVELODenoise(SPAVELO):
    def __init__(self, n_genes, true_time_switch, t_max, penalty_scale, log_variational, GP_dim, Normal_dim, encoder_layers, decoder_layers, encoder_dropout, decoder_dropout, 
                    fixed_inducing_points, initial_inducing_points, fixed_gp_params, kernel_scale, N_train, KL_loss, dynamicVAE, init_beta, min_beta, max_beta, KL_weight,
                    gamma_unconstr_init, alpha_unconstr_init, alpha_1_unconstr_init, lambda_alpha_unconstr_init, switch_spliced, switch_unspliced, linear_decoder, 
                    time_dep_transcription_rate, dirichlet_concentration, dtype, device, w_latent_loss=1, use_raw=False, adata=None, 
                    spliced_layer="spliced", unspliced_layer="unspliced", task="standard", use_rho=True):
        super(SPAVELODenoise, self).__init__(n_genes=n_genes, true_time_switch=true_time_switch, t_max=t_max, 
                                          penalty_scale=penalty_scale, log_variational=log_variational, GP_dim=GP_dim, 
                                          Normal_dim=Normal_dim, encoder_layers=encoder_layers, 
                                          decoder_layers=decoder_layers, encoder_dropout=encoder_dropout, decoder_dropout=decoder_dropout, 
                                          fixed_inducing_points=fixed_inducing_points, initial_inducing_points=initial_inducing_points, 
                                          fixed_gp_params=fixed_gp_params, kernel_scale=kernel_scale, N_train=N_train, KL_loss=KL_loss, 
                                          dynamicVAE=dynamicVAE, init_beta=init_beta, min_beta=min_beta, max_beta=max_beta, KL_weight=KL_weight,
                                          gamma_unconstr_init=gamma_unconstr_init, alpha_unconstr_init=alpha_unconstr_init, alpha_1_unconstr_init=alpha_1_unconstr_init, 
                                          lambda_alpha_unconstr_init=lambda_alpha_unconstr_init, switch_spliced=switch_spliced, switch_unspliced=switch_unspliced, linear_decoder=linear_decoder, 
                                          time_dep_transcription_rate=time_dep_transcription_rate, dirichlet_concentration=dirichlet_concentration, dtype=dtype, device=device, use_raw=use_raw, 
                                          adata=adata, spliced_layer=spliced_layer, unspliced_layer=unspliced_layer,
                                          task=task, use_rho=use_rho)
        self.w_latent_loss = w_latent_loss

    def decoder_generative(self, z, gamma, beta, alpha, alpha_1, lambda_alpha, latent_dim=None, y_spliced=None, y_unspliced=None):
        generative_outputs = super().decoder_generative(z, gamma, beta, alpha, alpha_1, lambda_alpha, latent_dim, y_spliced, y_unspliced)
        px_pi = generative_outputs["px_pi"]
        if not self.use_raw:
            mean_s = generative_outputs["mixture_dist_s"]._component_distribution.loc
            mean_u = generative_outputs["mixture_dist_u"]._component_distribution.loc
        else:
            mean_s = generative_outputs["mixture_dist_s"]._component_distribution.mu
            mean_u = generative_outputs["mixture_dist_u"]._component_distribution.mu
        fits_s = torch.sum(mean_s * px_pi, dim=-1)
        fits_u = torch.sum(mean_u * px_pi, dim=-1)
        generative_outputs["fits_s"] = fits_s
        generative_outputs["fits_u"] = fits_u
        return generative_outputs

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
        kl_pi = 0
        
        # 处理gamma_r2参数
        if gamma_r2 is not None:
            gamma_r2 = gamma_r2.reshape(1, -1) * torch.ones(size=(x.size(0),  y_spliced.size(1)), device=self.device)
            dynamo_mask = torch.where(gamma_r2 > 0, 1, 0)
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
            
            pred_s_1, pred_u_1 = generative_outputs["fits_s"], generative_outputs["fits_u"]
            encoder_output2 = self.encoder_inference(x, pred_s_1, pred_u_1)
            # p_m_2 = encoder_output2["p_m"]
            # p_v_2 = encoder_output2["p_v"]
            # latent_loss_mu = ((p_m - p_m_2).pow(2).sum())
            # latent_loss_var = ((p_v - p_v_2).pow(2).sum())
            # latent_loss = latent_loss_mu + latent_loss_var
            latent_loss = kl_divergence(encoder_output2["latent_dist"], latent_dist).sum()

            # print("p_m_shape", p_m_2.size(), p_m.size())
            reconst_loss += reconst_loss_u.sum() + reconst_loss_s.sum() + self.w_latent_loss * latent_loss

            # Use provided prior probabilities for Dirichlet KL divergence
            # Use the provided prior probabilities (must be provided)
            if prior_prob is None:
                raise ValueError("prior_prob must be provided when use_state_decoder=False")

            # kl_pi = F.mse_loss(px_pi.reshape(px_pi.shape[0], -1), torch.ones_like(prior_prob.reshape(prior_prob.shape[0], -1), device=self.device) * 0.25, reduction="sum").sum() / px_pi_alpha.shape[1]
            # kl_pi = F.mse_loss(px_pi.reshape(px_pi.shape[0], -1), prior_prob.reshape(prior_prob.shape[0], -1), reduction="sum").sum() / px_pi_alpha.shape[1]
            # kl_pi += F.cross_entropy(px_pi.reshape(-1, 4), prior_prob.argmax(dim=2).reshape(-1), reduction="sum").sum() / px_pi_alpha.shape[1]
            kl_pi_dynamo = kl_divergence(Dirichlet(px_pi_alpha), Dirichlet(torch.clamp(prior_prob, 1e-6, None))) * dynamo_mask
            # kl_pi_dynamo = kl_divergence(Dirichlet(px_pi_alpha), Dirichlet(torch.clamp(prior_prob, 1e-6, None))).sum()
            kl_pi_standard = kl_divergence(Dirichlet(px_pi_alpha), Dirichlet(torch.ones_like(prior_prob, device=self.device) * 0.25)) * standard_mask
            kl_pi = (self.KL_weight / px_pi_alpha.shape[1] * kl_pi_dynamo.sum() + kl_pi_standard.sum() / px_pi_alpha.shape[1])
            # kl_pi = self.KL_weight * kl_pi_dynamo.sum() / px_pi_alpha.shape[1]
            # kl_pi = kl_pi_dynamo * self.KL_weight / px_pi_alpha.shape[1]
            # kl_pi += (self.KL_weight / px_pi_alpha.shape[1]) * kl_pi_dynamo
        reconst_loss = reconst_loss / n_samples
        end_penalty_loss = end_penalty_loss / n_samples
        kl_pi = kl_pi / n_samples

        # ELBO
        penalty_weight = self._compute_penalty_weight(epoch)
        penalty_weight = torch.tensor(1., device=self.device)
        elbo = reconst_loss + kl_pi + self.penalty_scale * penalty_weight * end_penalty_loss + self.KL_beta * (kl_gp + kl_gaussian)

        return elbo, reconst_loss, kl_pi, end_penalty_loss, kl_gp, kl_gaussian, p_m, p_v, latent_samples

    @torch.inference_mode()
    def get_expression_fit_manual(self, adata, spatial_layer="spatial", spliced_layer="Ms", unspliced_layer="Mu", 
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

                fit_s, fit_u = self._compute_encoder_backinput(generative_outputs=generative_outputs)

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
