import numpy as np
import pandas as pd
import scipy.sparse
import scvelo as scv
from anndata import AnnData
from sklearn.preprocessing import MinMaxScaler
import torch
import scipy
from scvelo.tools.velocity import Velocity
import dynamo as dyn
import scanpy as sc


def preprocess_data(adata, spatial_layer="spatial", spliced_layer="Ms", unspliced_layer="Mu",
                    log_transform=True, min_max_scale=True, filter_on_r2=True, dtype="float64",
                    filter_on_ss=False):
    """Preprocess data.

    This function removes poorly detected genes and minmax scales the data.

    Parameters
    ----------
    adata
        Annotated data matrix.
    spliced_layer
        Name of the spliced layer.
    unspliced_layer
        Name of the unspliced layer.
    min_max_scale
        Min-max scale spliced and unspliced
    filter_on_r2
        Filter out genes according to linear regression fit

    Returns
    -------
    Preprocessed adata.
    """

#    adata.var_names_make_unique()
#    scv.pp.filter_and_normalize(adata, n_top_genes=2000)
    scv.pp.moments(adata)
#    if isinstance(adata.X, scipy.sparse.csr_matrix):
#        adata.X = np.array(adata.X.todense())
#    if isinstance(adata.layers[spliced_layer], scipy.sparse.csr_matrix):
#        adata.layers[spliced_layer] = np.array(adata.layers[spliced_layer].todense())
#    if isinstance(adata.layers[unspliced_layer], scipy.sparse.csr_matrix):
#        adata.layers[unspliced_layer] = np.array(adata.layers[unspliced_layer].todense())
    adata.layers[spliced_layer] = adata.layers[spliced_layer].astype(dtype)
    adata.layers[unspliced_layer] = adata.layers[unspliced_layer].astype(dtype)
    adata.obsm[spatial_layer] =  adata.obsm[spatial_layer].astype(dtype)


    if log_transform:
        adata.layers[spliced_layer] = np.log(adata.layers[spliced_layer]+1)
        adata.layers[unspliced_layer] = np.log(adata.layers[unspliced_layer]+1)

    if min_max_scale:
        scaler = MinMaxScaler()
        adata.layers["true_spliced"] = scaler.fit_transform(adata.layers["true_spliced"])

        scaler = MinMaxScaler()
        adata.layers["true_unspliced"] = scaler.fit_transform(adata.layers["true_unspliced"])

        scaler = MinMaxScaler()
        adata.layers[spliced_layer] = scaler.fit_transform(adata.layers[spliced_layer])

        scaler = MinMaxScaler()
        adata.layers[unspliced_layer] = scaler.fit_transform(adata.layers[unspliced_layer])

    if filter_on_r2:
        scv.tl.velocity(adata, mode="deterministic")

        adata = adata[
            :, np.logical_and(adata.var.velocity_r2 > 0, adata.var.velocity_gamma > 0)
        ].copy()
        adata = adata[:, adata.var.velocity_genes].copy()

    if filter_on_ss:
        velo = Velocity(adata, use_raw=False)
        velo.compute_deterministic(perc=[5, 95])
        var_names = adata.var_names[velo._velocity_genes]
        adata.var["fit_r2"] = velo._r2
        adata = adata[:, var_names].copy()

    return adata


def _softplus_inverse(x):
    x = torch.from_numpy(x)
    x_inv = torch.where(x > 20, x, x.expm1().log()).numpy()
    return x_inv


def get_init_variables(adata, spliced_layer="Ms", unspliced_layer="Mu", gamma_init_data=False, dynamo_init=False):
    spliced = adata.layers[spliced_layer]
    unspliced = adata.layers[unspliced_layer]

    sorted_unspliced = np.argsort(unspliced, axis=0)
    ind = int(adata.n_obs * 0.99)
    us_upper_ind = sorted_unspliced[ind:, :]

    us_upper = []
    ms_upper = []
    for i in range(len(us_upper_ind)):
        row = us_upper_ind[i]
        us_upper += [unspliced[row, np.arange(adata.n_vars)][np.newaxis, :]]
        ms_upper += [spliced[row, np.arange(adata.n_vars)][np.newaxis, :]]
    us_upper = np.median(np.concatenate(us_upper, axis=0), axis=0)
    ms_upper = np.median(np.concatenate(ms_upper, axis=0), axis=0)

    alpha_unconstr = _softplus_inverse(us_upper)
    alpha_unconstr = np.asarray(alpha_unconstr).ravel()

    alpha_1_unconstr = np.zeros(us_upper.shape).ravel()
    lambda_alpha_unconstr = np.zeros(us_upper.shape).ravel()

    if dynamo_init:
        adata.uns["pp"] = {'has_splicing': True,
                            'has_labeling': False,
                            'splicing_labeling': False,
                            'has_protein': False,
                            'tkey': None,
                            'experiment_type': 'conventional'}
        adata.var["pass_basic_filter"] = True
        adata.obs["pass_basic_filter"] = True
        adata.layers["X_spliced"] = adata.layers["spliced"]
        adata.layers["X_unspliced"] = adata.layers["unspliced"]
        dyn.tl.dynamics(adata)
        kinetics = pd.DataFrame(adata.varm["vel_params"], columns=adata.uns["vel_params_names"])
        dynamo_gamma_unconstr = (kinetics["gamma"] / kinetics["beta"]).values
        dynamo_gene_mask = ~np.isnan(dynamo_gamma_unconstr)
        gamma_r2 = np.where(~np.isnan(dynamo_gamma_unconstr), kinetics["gamma_r2"].values, -np.inf)

        adata = adata[:, dynamo_gene_mask].copy()
        us_upper = us_upper[dynamo_gene_mask]
        ms_upper = ms_upper[dynamo_gene_mask]
        alpha_unconstr = alpha_unconstr[dynamo_gene_mask]
        alpha_1_unconstr = alpha_1_unconstr[dynamo_gene_mask]
        lambda_alpha_unconstr = lambda_alpha_unconstr[dynamo_gene_mask]
        spliced = spliced[:, dynamo_gene_mask]
        unspliced = unspliced[:, dynamo_gene_mask]
        dynamo_gamma_unconstr = dynamo_gamma_unconstr[dynamo_gene_mask]
        steady_gamma_unconstr = dynamo_gamma_unconstr
        steady_beta_unconstr = np.ones_like(steady_gamma_unconstr)
        gamma_r2 = gamma_r2[dynamo_gene_mask]

        gamma_prime_init = (steady_gamma_unconstr / steady_beta_unconstr)
        s_lower, u_lower = np.array(0.05), np.array(0.05)
        s_upper, u_upper = np.quantile(adata.layers[spliced_layer], 0.9, axis=0), np.quantile(adata.layers[unspliced_layer], 0.9, axis=0) 
        velo = unspliced - gamma_prime_init * spliced
        upper_steady_mask = np.logical_and(spliced > s_upper, unspliced > u_upper)
        upper_steady_mask = np.logical_and(upper_steady_mask, np.abs(velo) < 0.1)
        lower_steady_mask = np.logical_and(spliced < s_lower, unspliced < u_lower)
        lower_steady_mask = np.logical_and(lower_steady_mask, np.abs(velo) < 0.1)
        non_steady_mask = np.logical_not(np.logical_or(upper_steady_mask, lower_steady_mask))
        ind_prob = np.where(np.logical_and(velo >= 0, non_steady_mask), 1, 0)
        ind_steady_prob = np.where(upper_steady_mask, 1, 0)
        rep_prob = np.where(np.logical_and(velo < 0, non_steady_mask), 1, 0)
        rep_steady_prob = np.where(lower_steady_mask, 1, 0)
        prior_prob = np.stack([ind_prob, ind_steady_prob, rep_prob, rep_steady_prob], axis=-1)
    else:
        ind_prob = np.ones(adata.shape) * 0.25
        ind_steady_prob = np.ones(adata.shape) * 0.25
        rep_prob = np.ones(adata.shape) * 0.25
        rep_steady_prob = np.ones(adata.shape) * 0.25
        prior_prob = np.stack([ind_prob, ind_steady_prob, rep_prob, rep_steady_prob], axis=-1)
        steady_beta_unconstr = None
        steady_gamma_unconstr = None
        gamma_r2 = np.ones(adata.shape[1])

    if gamma_init_data:
        gamma_unconstr = np.clip(_softplus_inverse(us_upper / ms_upper), 1e-10, 10)
        gamma_unconstr = steady_gamma_unconstr.copy()
    else:
        gamma_unconstr = None

    return adata, us_upper, ms_upper, alpha_unconstr, gamma_unconstr, steady_beta_unconstr, steady_gamma_unconstr, alpha_1_unconstr, lambda_alpha_unconstr, prior_prob, gamma_r2
