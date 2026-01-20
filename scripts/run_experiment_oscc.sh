#!/bin/bash
set -e
cd ../experiments
nohup python -u run_spaVelo_real_deno.py --data_file /data/chensishuo/spvelo_data/real_data/oscc/merged/sample_2/data.h5ad \
            --inducing_point_steps 15 --gene_selection spatial --device cuda:5 --w_latent_loss 1 --KL_weight 10 --penalty_scale 10 > ../output/s2/spavelo_deno_oscc_s2_$(date +%Y%m%d_%H%M%S).txt &
sleep 10
nohup python -u run_spaVelo_real_deno.py --data_file /data/chensishuo/spvelo_data/real_data/oscc/merged/sample_5/data.h5ad \
            --inducing_point_steps 15 --gene_selection spatial --device cuda:5 --w_latent_loss 1 --KL_weight 10 --penalty_scale 10 > ../output/s5/spavelo_deno_oscc_s5_$(date +%Y%m%d_%H%M%S).txt &
sleep 10
nohup python -u run_spaVelo_real_deno.py --data_file /data/chensishuo/spvelo_data/real_data/oscc/merged/sample_7/data.h5ad \
            --inducing_point_steps 15 --gene_selection spatial --device cuda:5 --w_latent_loss 1 --KL_weight 10 --penalty_scale 10 > ../output/s7/spavelo_deno_oscc_s7_$(date +%Y%m%d_%H%M%S).txt &
sleep 10
nohup python -u run_spaVelo_real_deno.py --data_file /data/chensishuo/spvelo_data/real_data/oscc/merged/sample_8/data.h5ad \
            --inducing_point_steps 15 --gene_selection spatial --device cuda:5 --w_latent_loss 1 --KL_weight 10 --penalty_scale 10 > ../output/s8/spavelo_deno_oscc_s8_$(date +%Y%m%d_%H%M%S).txt &
sleep 10
nohup python -u run_spaVelo_real_deno.py --data_file /data/chensishuo/spvelo_data/real_data/oscc/merged/sample_9/data.h5ad \
            --inducing_point_steps 15 --gene_selection spatial --device cuda:5 --w_latent_loss 1 --KL_weight 10 --penalty_scale 10 > ../output/s9/spavelo_deno_oscc_s9_$(date +%Y%m%d_%H%M%S).txt &
sleep 10
nohup python -u run_spaVelo_real_deno.py --data_file /data/chensishuo/spvelo_data/real_data/oscc/merged/sample_10/data.h5ad \
            --inducing_point_steps 15 --gene_selection spatial --device cuda:5 --w_latent_loss 1 --KL_weight 10 --penalty_scale 10 > ../output/s10/spavelo_deno_oscc_s10_$(date +%Y%m%d_%H%M%S).txt &
sleep 10
cd -
set +e
