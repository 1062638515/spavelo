import math, os
import time
import sys
sys.path.append("..")

import torch
from spaVelo import SPAVELO
from spaVeloDenoise import SPAVELODenoise
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import h5py
import scanpy as sc
from preprocess_real import preprocess_data_real, get_init_variables

import pandas as pd
import scvelo as scv
import seaborn as sns
import matplotlib.pyplot as plt

import torch.nn.functional as F

from tools.norm_vector import neighbors
import random

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# torch.manual_seed(42)

def draw_velocity_stream(adata, spatial=False, color=None, xkey="pred_s", vkey="pred_velocity"):
    n_color = len(adata.obs[color].unique())
    if not spatial:
        scv.pp.neighbors(adata)
        # neighbors(adata, use_rep="X_pca")
        scv.tl.velocity_graph(adata, vkey=vkey, xkey=xkey)
        scv.pl.velocity_embedding_stream(adata, color=color, vkey=vkey, palette=sns.color_palette("hls", n_color, 1), size=100, basis="umap", legend_loc="right", alpha=1)
    else:
        scv.pp.neighbors(adata, use_rep="X_coord")
        # neighbors(adata, use_rep="X_coord")
        scv.tl.velocity_graph(adata, vkey=vkey, xkey=xkey)
        scv.pl.velocity_embedding_stream(adata, color=color, vkey=vkey, palette=sns.color_palette("hls", n_color, 1), basis="X_coord", size=100, legend_loc="right", alpha=1)
    plt.tight_layout()

if __name__ == "__main__":

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='Spatial dependency-aware variational autoencoder for RNA velocity analysis',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_file', default='data_simu.h5ad')
    parser.add_argument('--spliced_layer', type=str, default='spliced')
    parser.add_argument('--unspliced_layer', type=str, default='unspliced')
    parser.add_argument('--batch_size', default=256)
    parser.add_argument('--maxiter', default=5000, type=int)
    parser.add_argument('--train_size', default=0.95, type=float)
    parser.add_argument('--patience', default=100, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--dropoutE', default=0., type=float,
                        help='dropout probability for encoder')
    parser.add_argument('--dropoutD', default=0., type=float,
                        help='dropout probability for decoder')
    parser.add_argument('--encoder_layers', nargs="+", default=[256, 128], type=int)
    parser.add_argument('--GP_dim', default=2, type=int,help='dimension of the latent Gaussian process embedding')
    parser.add_argument('--Normal_dim', default=8, type=int,help='dimension of the latent standard Gaussian embedding')
    parser.add_argument('--decoder_layers', nargs="+", default=[128, 256], type=int)
    parser.add_argument('--dynamicVAE', default=True, type=bool, 
                        help='whether to use dynamicVAE to tune the value of beta, if setting to false, then beta is fixed to initial value')
    parser.add_argument('--init_beta', default=4, type=float, help='initial coefficient of the KL loss')
    parser.add_argument('--min_beta', default=1, type=float, help='minimal coefficient of the KL loss')
    parser.add_argument('--max_beta', default=25, type=float, help='maximal coefficient of the KL loss')
    parser.add_argument('--KL_loss', default=0.025, type=float, help='desired KL_divergence value')
    parser.add_argument('--KL_weight', default=1., type=float, help='weight of Dirichlet KL_divergence')
    parser.add_argument('--penalty_scale', default=0.2, type=float, help='coefficient of the penalty')
    parser.add_argument('--n_samples', default=1, type=int)
    parser.add_argument('--fix_inducing_points', default=True, type=bool)
    parser.add_argument('--grid_inducing_points', default=True, type=bool, 
                        help='whether to generate grid inducing points or use k-means centroids on locations as inducing points')
    parser.add_argument('--inducing_point_steps', default=15, type=int)
    parser.add_argument('--inducing_point_nums', default=None, type=int)
    parser.add_argument('--fixed_gp_params', default=False, type=bool)
    parser.add_argument('--loc_range', default=20., type=float)
    parser.add_argument('--w_latent_loss', default=1., type=float)
    parser.add_argument('--kernel_scale', default=20., type=float)
    parser.add_argument('--gene_selection', default="spatial", type=str)
    parser.add_argument('--use_raw', action="store_true")
    parser.add_argument('--no_rho', action="store_true")
    parser.add_argument('--save_dir', default="../logs/real_data/" + time.strftime("%Y%m%d-%H%M%S", time.localtime()))
    parser.add_argument('--model_file', default="model.pt")
    parser.add_argument('--device', default='cuda:0')


    args = parser.parse_args()

    set_seed(144)
    os.makedirs(args.save_dir, exist_ok=True)
    args.model_file = f"{args.save_dir}/{args.model_file}"
    pd.Series(vars(args)).to_csv(f"{args.save_dir}/args.csv")
    adata = sc.read_h5ad(args.data_file)

    if args.batch_size == "auto":
        if adata.shape[0] <= 1024:
            args.batch_size = 128
        elif adata.shape[0] <= 2048:
            args.batch_size = 256
        else:
            args.batch_size = 512
    else:
        args.batch_size = int(args.batch_size)

    print(args)

    adata.obsm["X_coord_raw"] = adata.obsm["X_coord"].astype(np.float64)
    scaler = MinMaxScaler()
    adata.obsm["X_coord"] = scaler.fit_transform(adata.obsm["X_coord"]) * args.loc_range

    print(adata.shape)
    print(adata.obsm["X_coord"].shape)
    # if "spatial" not in adata.obsm:
    adata.obsm["spatial"] = adata.obsm["X_coord"]

    # We provide two ways to generate inducing point, argument "grid_inducing_points" controls whether to choice grid inducing or k-means
    # One way is grid inducing points, argument "inducing_point_steps" controls number of grid steps, the resulting number of inducing point is (inducing_point_steps+1)^2
    # Another way is k-means on the locations, argument "inducing_point_nums" controls number of inducing points
    if args.grid_inducing_points:
        eps = 1e-5
        initial_inducing_points = np.mgrid[0:(1+eps):(1./args.inducing_point_steps), 0:(1+eps):(1./args.inducing_point_steps)].reshape(2, -1).T * args.loc_range
        print(initial_inducing_points.shape)
    else:
        loc = adata.obsm["X_coord"]
        loc_kmeans = KMeans(n_clusters=args.inducing_point_nums, n_init=100).fit(loc)
        np.savetxt("location_centroids.txt", loc_kmeans.cluster_centers_, delimiter=",")
        np.savetxt("location_kmeans_labels.txt", loc_kmeans.labels_, delimiter=",", fmt="%i")
        initial_inducing_points = loc_kmeans.cluster_centers_

    adata = preprocess_data_real(adata, spatial_layer="X_coord", spliced_layer=args.spliced_layer, unspliced_layer=args.unspliced_layer, dtype="float64", log_transform=True, min_max_scale=True, filter_on_r2=False, filter_on_ss=False, gene_selection=args.gene_selection, use_raw=args.use_raw, smooth_pca=False)
    adata, us_upper, ms_upper, alpha_unconstr, gamma_unconstr, steady_beta_unconstr, steady_gamma_unconstr, alpha_1_unconstr, lambda_alpha_unconstr, prior_prob, gamma_r2 = get_init_variables(adata, spliced_layer=args.spliced_layer, unspliced_layer=args.unspliced_layer, gamma_init_data=False, dynamo_init=True)
    adata.layers["prior_prob"] = prior_prob.argmax(axis=2)

    model = SPAVELODenoise(n_genes=adata.n_vars, true_time_switch=None, t_max=20, penalty_scale=args.penalty_scale, log_variational=False,
                   GP_dim=args.GP_dim, Normal_dim=args.Normal_dim, encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers,
                   encoder_dropout=args.dropoutE, decoder_dropout=args.dropoutD,
                   fixed_inducing_points=args.fix_inducing_points, initial_inducing_points=initial_inducing_points, 
                   fixed_gp_params=args.fixed_gp_params, kernel_scale=args.kernel_scale, N_train=adata.n_obs, KL_loss=args.KL_loss, dynamicVAE=args.dynamicVAE, 
                   init_beta=args.init_beta, min_beta=args.min_beta, max_beta=args.max_beta, KL_weight=args.KL_weight,
                   gamma_unconstr_init=gamma_unconstr, alpha_unconstr_init=alpha_unconstr, alpha_1_unconstr_init=alpha_1_unconstr,
                   lambda_alpha_unconstr_init=lambda_alpha_unconstr, switch_spliced=ms_upper, switch_unspliced=us_upper, linear_decoder=False, 
                   time_dep_transcription_rate=False, dirichlet_concentration=0.25, dtype=torch.float64, device=args.device, w_latent_loss=args.w_latent_loss, 
                   use_raw=args.use_raw, adata=adata, spliced_layer=args.spliced_layer, unspliced_layer=args.unspliced_layer,
                   use_rho=(not args.no_rho))

    print(str(model))

    if not os.path.isfile(args.model_file):
        t0 = time.time()
        model.train_model(adata, prior_prob=prior_prob, spatial_layer="X_coord", spliced_layer=args.spliced_layer, unspliced_layer=args.unspliced_layer,
                lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size, n_samples=args.n_samples,
                train_size=args.train_size, maxiter=args.maxiter, patience=args.patience, save_model=True, model_weights=args.model_file, log_dir=args.save_dir, gamma_r2=gamma_r2)
        print('Training time: %d seconds.' % int(time.time() - t0))
    else:
        model.load_model(args.model_file)

    # model.load_model(args.model_file)
    # model.eval()
    latent_z = model.get_encoder_latent(adata, spatial_layer="X_coord", spliced_layer=args.spliced_layer, unspliced_layer=args.unspliced_layer,
                                                   batch_size=args.batch_size, n_samples=25)
    adata.obsm["latent_z"] = latent_z
    pred_t = model.get_latent_time(adata, spatial_layer="X_coord", spliced_layer=args.spliced_layer, unspliced_layer=args.unspliced_layer,
                                                   batch_size=args.batch_size, n_samples=25)
    adata.varm.update(model.get_rates())
    adata.layers["pred_t"] = pred_t
    adata.obs["pred_t"] = pred_t.mean(axis=1)
    adata.layers["fit_t"] = adata.layers["pred_t"]
    # scv.pp.neighbors(adata, use_rep="X_coord")
    # scv.tl.velocity_graph(adata, vkey="pred_velocity")
    # scv.tl.latent_time(adata, vkey="pred_velocity", min_likelihood=None)
    # adata.layers["state"] = model.get_state_assignment(adata, spatial_layer="X_coord", spliced_layer="spliced", unspliced_layer="unspliced", batch_size=args.batch_size, n_samples=25)
    adata.layers["pred_velocity"] = model.get_velocity(adata, spatial_layer="X_coord", spliced_layer=args.spliced_layer, unspliced_layer=args.unspliced_layer,
                                                       batch_size=args.batch_size, n_samples=25)
    
    # 计算细胞级别的latent time
    print("Computing cell-level latent time...")
    cell_latent_time = model.get_cell_latent_time(adata, group_key="Cluster", spatial_layer="X_coord", spliced_layer=args.spliced_layer, 
                                                   unspliced_layer=args.unspliced_layer, vkey="pred_velocity",
                                                   leiden_resolution=0.5, batch_size=args.batch_size, n_samples=25)
    switches = ([0.4, 0.7, 1, 0.1] * 500)[:1000]
    # draw_gene_level_plot(adata, switches=switches, pred_alpha=adata.varm["alpha"],
    #                     true_alpha=adata.uns["alpha"], true_t_key="true_t", pred_t_key="pred_t",
    #                     save_dir=args.save_dir)
    state = model.get_state_assignment(adata, spatial_layer="X_coord", spliced_layer=args.spliced_layer, unspliced_layer=args.unspliced_layer,
                                       batch_size=args.batch_size, n_samples=25)
    adata.layers["state_prob"] = state[0]
    adata.layers["state"] = state[0].argmax(axis=2)
    adata.uns["state_order"] = state[1]
    rho = model.get_rho(adata, spatial_layer="X_coord", spliced_layer=args.spliced_layer, unspliced_layer=args.unspliced_layer,
                        batch_size=args.batch_size, n_samples=25)
    adata.layers["rho"] = rho
    expression_fit = model.get_expression_fit(adata, spatial_layer="X_coord", spliced_layer=args.spliced_layer, unspliced_layer=args.unspliced_layer,
                                              batch_size=args.batch_size, n_samples=25)
    adata.layers["pred_s"] = expression_fit[0]
    adata.layers["pred_u"] = expression_fit[1]
    gene_likelihood = model.get_gene_likelihood(adata, spatial_layer="X_coord", spliced_layer=args.spliced_layer, unspliced_layer=args.unspliced_layer,
                                            batch_size=args.batch_size, n_samples=25)
    adata.layers["likelihood"] = gene_likelihood

    adata.var["switch_pred"] = F.softplus(model.switch_time_unconstr).detach().cpu().numpy()

    adata.obsm["X_coord"] = adata.obsm["X_coord_raw"]
    sc.write(f"{args.save_dir}/spavelo_adata.h5ad", adata)
    # adata.obsm["X_coord"][:, 1] = adata.obsm["X_coord"][:, 1] * (-1)
    draw_velocity_stream(adata.copy(), color="Cluster", xkey="Ms", vkey="pred_velocity")
    plt.savefig(f"{args.save_dir}/velo_umap.png", dpi=600)
    draw_velocity_stream(adata.copy(), spatial=True, color="Cluster", xkey="Ms", vkey="pred_velocity")
    plt.savefig(f"{args.save_dir}/velo_spatial.png", dpi=600)
    adata.obs["x"] = adata.obsm["X_coord"][:, 0]
    adata.obs["y"] = adata.obsm["X_coord"][:, 1]
    sc.pl.scatter(adata, x="x", y="y", color="pred_t", show=False, color_map="gnuplot", size=70)
    plt.savefig(f"{args.save_dir}/pred_t.png", dpi=600)
    sc.pl.scatter(adata, x="x", y="y", color="cell_latent_time", show=False, color_map="gnuplot", size=70)
    plt.savefig(f"{args.save_dir}/cell_latent_time.png", dpi=600)

# radial
# nohup python run_spaVelo.py --data_file ../../simulate/result/radial/pos_noise/1/20240508_063130/data_simu.h5ad --device cuda:5 --inducing_point_steps 15 > output/spavelo_20240508_063130.txt &
# bidirect
# nohup python run_spaVelo.py --data_file ../../simulate/result/bidirect/exp_noise/1.1/20240512_122022/data_simu.h5ad --device cuda:0 --inducing_point_steps 10 > output/spavelo_20240512_122022.txt &
