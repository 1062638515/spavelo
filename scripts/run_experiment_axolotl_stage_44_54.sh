#!/bin/bash
set -e
cd ../experiments
nohup python -u run_spaVelo_real_deno.py --data_file /HDD1/chensishuo/spvelo_data/real_data/stereo/axolotl/processed/Stage44_telencephalon_rep2_FP200000239BL_E4_WMPZY/processed.h5ad \
            --inducing_point_steps 15 --gene_selection spatial --device cuda:2 --w_latent_loss 1 --KL_weight 10 --penalty_scale 10 > ../output/stage44/spavelo_deno_axolotl_stage44_$(date +%Y%m%d%H%M%S).txt &
sleep 5
nohup python -u run_spaVelo_real_deno.py --data_file /HDD1/chensishuo/spvelo_data/real_data/stereo/axolotl/processed/Stage54_telencephalon_rep2_DP8400015649BRD6_2_JZYLH/processed.h5ad \
            --inducing_point_steps 15 --gene_selection spatial --device cuda:2 --w_latent_loss 1 --KL_weight 10 --penalty_scale 10 > ../output/stage54/spavelo_deno_axolotl_stage54_$(date +%Y%m%d%H%M%S).txt &
sleep 5
cd -
set +e
