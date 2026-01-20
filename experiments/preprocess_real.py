import numpy as np
import pandas as pd
import scipy.sparse
import scvelo as scv
from anndata import AnnData
from sklearn.preprocessing import MinMaxScaler
import torch
import scipy
from scvelo.tools.velocity import Velocity
import scanpy as sc
import squidpy as sq
from scipy.sparse import issparse
import dynamo as dyn


class UnitVeloGeneSelector:
    def __init__(self, ):
        self.min_ratio = 0.01
        self.min_r2 = 0.01
        self.R2_ADJUST = True
        self.gene_index = None
        self.r2 = None
        self.gamma_ref = None
        self.residual_scale = None
        self.gamma_quantile = None
        self.scaling = None
        self.nobs = None

    def get_weight(self, x, y=None, perc=95):
        xy_norm = np.array(x.A if issparse(x) else x)
        if y is not None:
            if issparse(y):
                y = y.A
            xy_norm = xy_norm / np.clip(np.max(xy_norm, axis=0), 1e-3, None)
            xy_norm += y / np.clip(np.max(y, axis=0), 1e-3, None)

        if isinstance(perc, int):
            weights = xy_norm >= np.percentile(xy_norm, perc, axis=0)
        else:
            lb, ub = np.percentile(xy_norm, perc, axis=0)
            weights = (xy_norm <= lb) | (xy_norm >= ub)

        return weights

    def R2(self, residual, total):
        r2 = np.ones(residual.shape[1]) - \
             np.sum(residual * residual, axis=0) / \
             np.sum(total * total, axis=0)
        r2[np.isnan(r2)] = 0
        return r2

    def init_weights(self, spliced, unspliced):
        nonzero_s, nonzero_u = spliced > 0, unspliced > 0
        weights = np.array(nonzero_s & nonzero_u, dtype=bool)
        self.nobs = np.sum(weights, axis=0)

    def get_genes(self, adata:sc.AnnData):
        variable = adata.var
        if variable.index[0].startswith('ENSMUSG'):
            variable.index = variable['gene']
            variable.index.name = 'index' 
        
        weights = self.get_weight(adata.layers["spliced"], adata.layers["unspliced"], perc=95)
        spliced, unspliced = weights * adata.layers["spliced"].toarray(), weights * adata.layers["unspliced"].toarray()

        self.gamma_quantile = np.sum(unspliced * spliced, axis=0) / np.sum(spliced * spliced, axis=0)
        self.scaling = np.std(unspliced, axis=0) / np.std(spliced, axis=0)
        adata.layers['Mu_scale'] = unspliced / self.scaling

        self.gene_index = variable.index
        self.gamma_ref = np.sum(unspliced * spliced, axis=0) / np.sum(spliced * spliced, axis=0)
        self.residual_scale = unspliced - self.gamma_ref * spliced
        self.r2 = self.R2(self.residual_scale, total=unspliced - np.mean(unspliced, axis=0))

        self.velocity_genes = np.ones(spliced.shape[1], dtype=bool)

        self.velocity_genes = (
            (self.r2 > self.min_r2)
            & (self.r2 < 0.95)
            & (self.gamma_quantile > self.min_ratio)
            & (self.gamma_ref > self.min_ratio)
            & (np.max(adata.layers["spliced"].toarray() > 0, axis=0) > 0)
            & (np.max(adata.layers["unspliced"].toarray() > 0, axis=0) > 0)
        )
        print (f'# of velocity genes {self.velocity_genes.sum()} (Criterion: positive regression coefficient between un/spliced counts)')
        
        if self.R2_ADJUST and False:
            lb, ub = np.nanpercentile(self.scaling, [10, 90])
            self.velocity_genes = (
                self.velocity_genes
                & (self.scaling > np.min([lb, 0.03]))
                & (self.scaling < np.max([ub, 3]))
            )
        print (f'# of velocity genes {self.velocity_genes.sum()} (Criterion: std of un/spliced reads should be moderate, w/o extreme values)')

        self.init_weights(spliced=adata.layers["spliced"].toarray(), unspliced=adata.layers["unspliced"].toarray())
        self.velocity_genes = adata[:, self.velocity_genes & (self.nobs > 0.01 * spliced.shape[1])].var.index.to_list()

        if len(self.velocity_genes) < 2:
            print ('---> Low signal in splicing dynamics.')
        return self.velocity_genes



def preprocess_data_real(adata, spatial_layer="spatial", spliced_layer="Ms", unspliced_layer="Mu",
                         log_transform=True, min_max_scale=True, filter_on_r2=True, dtype="float64",
                         filter_on_ss=False, gene_selection="spatial_hvg", use_raw=False, smooth_pca=False):
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

    adata.var_names_make_unique()
    adata_raw = adata.copy()
    # unitvelo_gene_selector = UnitVeloGeneSelector()
    # utv_genes = unitvelo_gene_selector.get_genes(adata)
    # adata = adata[:, utv_genes].copy()
    # scv.pp.filter_and_normalize(adata, n_top_genes=2000)
    if gene_selection == "spatial_hvg":
        adata_moran = adata.copy()
        adata_hvg = adata.copy()
        sq.gr.spatial_neighbors(adata_moran, n_neighs=8)
        expressed_genes = (adata_moran.layers["spliced"].A + adata_moran.layers["unspliced"].A).sum(axis=0) > 500
        expressed_genes = adata_moran.var_names[expressed_genes]
        sc.pp.normalize_total(adata_moran)
        sc.pp.log1p(adata_moran)
        sq.gr.spatial_autocorr(adata_moran, mode="moran",  genes=list(expressed_genes), n_perms=100, n_jobs=30)
        scv.pp.filter_and_normalize(adata_hvg, n_top_genes=3000)
        hvg_names = adata_hvg.var.index
        moran_top_gene = adata_moran.uns["moranI"].head(2000).index
        final_gene = sorted(set(moran_top_gene) & set(expressed_genes) & set(hvg_names))
        adata = adata[:, list(final_gene)].copy()
        print(adata.layers["spliced"].todense())
        if not use_raw:
            scv.pp.filter_and_normalize(adata)
    elif gene_selection == "spatial_scv":
        adata_moran = adata.copy()
        sq.gr.spatial_neighbors(adata_moran, n_neighs=8)
        expressed_genes = (adata_moran.layers["spliced"].A + adata_moran.layers["unspliced"].A).sum(axis=0) > 500
        expressed_genes = adata_moran.var_names[expressed_genes]
        sc.pp.normalize_total(adata_moran)
        sc.pp.log1p(adata_moran)
        sq.gr.spatial_autocorr(adata_moran, mode="moran",  genes=list(expressed_genes), n_perms=100, n_jobs=30)
        moran_top_gene = adata_moran.uns["moranI"].head(2000).index
        spatial_gene = list(set(moran_top_gene) & set(expressed_genes))
        adata_scv = adata[:, spatial_gene].copy()
        scv.pp.filter_and_normalize(adata_scv)
        scv.pp.moments(adata_scv)
        scv.tl.velocity(adata_scv)
        final_gene = sorted(list(adata_scv[:, adata_scv.var["velocity_genes"]].var.index))
        adata = adata[:, final_gene].copy()
        if not use_raw:
            scv.pp.filter_and_normalize(adata)
    elif gene_selection == "spatial_unitvelo":
        adata_raw = adata.copy()
        adata_moran = adata.copy()
        sq.gr.spatial_neighbors(adata_moran, n_neighs=8)
        expressed_genes = (adata_moran.layers["spliced"].A + adata_moran.layers["unspliced"].A).sum(axis=0) > 500
        expressed_genes = adata_moran.var_names[expressed_genes]
        sc.pp.normalize_total(adata_moran)
        sc.pp.log1p(adata_moran)
        sq.gr.spatial_autocorr(adata_moran, mode="moran",  genes=list(expressed_genes), n_perms=100, n_jobs=30)
        moran_top_gene = adata_moran.uns["moranI"].head(2000).index
        spatial_gene = list(set(moran_top_gene) & set(expressed_genes))
        adata_scv = adata[:, spatial_gene].copy()
        scv.pp.filter_and_normalize(adata_scv)
        scv.pp.moments(adata_scv)
        scv.tl.velocity(adata_scv)
        final_gene = sorted(list(adata_scv[:, adata_scv.var["velocity_genes"]].var.index))
        adata = adata[:, final_gene].copy()
        scv.pp.filter_and_normalize(adata)
        unitvelo_gene_selector = UnitVeloGeneSelector()
        utv_genes = unitvelo_gene_selector.get_genes(adata)
        adata = adata_raw[:, utv_genes]
        if not use_raw:
            scv.pp.filter_and_normalize(adata)
    elif gene_selection == "spatial":
        adata_moran = adata.copy()
        sq.gr.spatial_neighbors(adata_moran, n_neighs=8)
        expressed_genes = (adata_moran.layers["spliced"].A + adata_moran.layers["unspliced"].A).sum(axis=0) > 500
        expressed_genes = adata_moran.var_names[expressed_genes]
        sc.pp.normalize_total(adata_moran)
        sc.pp.log1p(adata_moran)
        sq.gr.spatial_autocorr(adata_moran, mode="moran",  genes=list(expressed_genes), n_perms=100, n_jobs=30)
        moran_top_gene = adata_moran.uns["moranI"].head(2000).index
        final_gene = sorted(list(set(moran_top_gene) & set(expressed_genes)))
        adata = adata[:, final_gene].copy()
        print("slected adata", adata.shape)
        if not use_raw:
            scv.pp.filter_and_normalize(adata)
    elif gene_selection == "spatial_full":
        adata_moran = adata.copy()
        u_steady_thresh = np.quantile(adata.layers["unspliced"].A, 0.8, axis=0)
        s_steady_thresh = np.quantile(adata.layers["spliced"].A, 0.8, axis=0)
        steady_genes = set(adata.var[np.logical_and(adata.layers["spliced"].A > s_steady_thresh, 
                            adata.layers["unspliced"].A > u_steady_thresh).sum(axis=0) > 0.1 * adata.shape[0]].index)
        adata_state = adata.copy()
        scv.pp.filter_and_normalize(adata_state, n_top_genes=10000)
        scv.tl.velocity(adata_state)
        state_mask = adata_state.layers["unspliced"].A - adata_state.var["velocity_gamma"].values.reshape(1, -1) * adata_state.layers["spliced"].A
        rep_count = (state_mask < 0).sum(axis=0)
        ind_count = (state_mask > 0).sum(axis=0)
        complete_genes = adata_state.var.index[np.logical_and(rep_count > 0.1 * adata.shape[0], ind_count > 0.1 * adata.shape[0])].to_list()
        sq.gr.spatial_neighbors(adata_moran, n_neighs=8)
        expressed_genes = (adata_moran.layers["spliced"].A + adata_moran.layers["unspliced"].A).sum(axis=0) > 500
        expressed_genes = adata_moran.var_names[expressed_genes]
        sc.pp.normalize_total(adata_moran)
        sc.pp.log1p(adata_moran)
        sq.gr.spatial_autocorr(adata_moran, mode="moran",  genes=list(expressed_genes), n_perms=100, n_jobs=30)
        moran_top_gene = adata_moran.uns["moranI"].head(2000).index
        final_gene = sorted(list(set(moran_top_gene) & set(expressed_genes) & steady_genes & set(complete_genes)))
        adata = adata[:, final_gene].copy()
        print("slected adata", adata.shape)
        if not use_raw:
            scv.pp.filter_and_normalize(adata)
    elif gene_selection == "none":
        if not use_raw:
            scv.pp.filter_and_normalize(adata)
    else:
        if not use_raw:
            scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    if not use_raw:
        if smooth_pca:
            raise NotImplementedError("RASP not supported")
        else:
            sc.pp.pca(adata)
            scv.pp.moments(adata)
    if isinstance(adata.X, scipy.sparse.csr_matrix):
        adata.X = np.array(adata.X.todense())
    if isinstance(adata.layers[spliced_layer], scipy.sparse.csr_matrix):
        adata.layers[spliced_layer] = np.array(adata.layers[spliced_layer].todense())
    if isinstance(adata.layers[unspliced_layer], scipy.sparse.csr_matrix):
        adata.layers[unspliced_layer] = np.array(adata.layers[unspliced_layer].todense())
    adata.layers[spliced_layer] = adata.layers[spliced_layer].astype(dtype)
    adata.layers[unspliced_layer] = adata.layers[unspliced_layer].astype(dtype)
    adata.obsm[spatial_layer] =  adata.obsm[spatial_layer].astype(dtype)


    if not use_raw and log_transform:
        adata.layers[spliced_layer] = np.log(adata.layers[spliced_layer]+1)
        adata.layers[unspliced_layer] = np.log(adata.layers[unspliced_layer]+1)

    if not use_raw and min_max_scale:
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
    adata_umap = adata_raw.copy()
    # if not use_raw:
        # scv.pp.filter_and_normalize(adata_umap)
    adata.obsm["X_umap"] = get_umap(adata_umap, adata.obs.index)

    return adata


def get_umap(adata, index=None):
    adata = adata.copy()
    scv.pp.filter_and_normalize(adata, n_top_genes=2000)
    if isinstance(adata.X, scipy.sparse.csr_matrix):
        adata.X = adata.X.toarray()
    sc.pp.pca(adata, svd_solver="full")
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    adata = adata[index]
    return adata.obsm["X_umap"]


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
        # dynamo_gene_mask = np.logical_and(~np.isnan(dynamo_gamma_unconstr), kinetics["gamma_r2"] > 0)
        dynamo_gene_mask = ~np.isnan(dynamo_gamma_unconstr)
        gamma_r2 = np.where(~np.isnan(dynamo_gamma_unconstr), kinetics["gamma_r2"].values, -np.inf)
        gamma_r2 = gamma_r2[dynamo_gene_mask]

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
