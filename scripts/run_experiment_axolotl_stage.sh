#!/bin/bash
set -e
cd ../experiments
nohup python -u run_spaVelo_real_deno.py --data_file /HDD1/chensishuo/spvelo_data/real_data/stereo/axolotl/processed/Stage57_telencephalon_rep2_DP8400015649BRD5_1_K0AP4/processed.h5ad \
            --inducing_point_steps 15 --gene_selection spatial --device cuda:4 --w_latent_loss 1 --KL_weight 15 --penalty_scale 10 > ../output/stage57/spavelo_deno_axolotl_stage57_$(date +%Y%m%d_%H%M%S).txt &
sleep 5
cd -
set +e
