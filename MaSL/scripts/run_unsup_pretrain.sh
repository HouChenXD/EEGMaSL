export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MASTER_ADDR=localhost
export MASTER_PORT=1234
export CUDA_VISIBLE_DEVICES="0" 
export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1

python /home/replace/EEG/code/BIOT/unsupervised_pretrain.py \
    --batch_size 512 \
    --epochs 100 \
    --num_workers 16 \
    --shhs_path "/home/replace/EEG/data/SHHS/processed" \
    --TUHseries_path "/mnt/replace_disk/EEG_data/TUHseries" \
    --save_path "./log-pretrain/unsupervised/shhs_tu_checkpoints_mamba2_freqmix" \
    --dropout 0.2