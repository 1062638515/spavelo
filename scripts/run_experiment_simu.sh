#!/bin/bash
set -e
cd ..
device="cuda:3"
function run_simu
{
    sample_name=${1%/*}
    sample_name=${sample_name##*/}
    if [ ! -d output/spavelo_simu_new/${sample_name} ]
    then
        mkdir output/spavelo_simu_new/${sample_name}
    fi
    nohup python run_spaVelo.py --data_file $1 \
                                --device ${device} --inducing_point_steps 15 --w_latent_loss 1 --KL_weight 10 --penalty_scale 10 > output/spavelo_simu_new/${sample_name}/$(date +%Y%m%d_%H%M%S).txt 2>&1 &
    sleep 5
}

group_dir=/home/chensishuo/spvelo/simulate/result/new/sectorial/exp_noise/0.5
files=$(ls ${group_dir})
i=0
for curr_sample in ${files}
do
    i=$(( i + 1 ))
    echo "sample_${i}: ${curr_sample}"
    run_simu ${group_dir}/${curr_sample}/data_simu.h5ad
done

cd -
set +e
