export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MASTER_ADDR=localhost
export MASTER_PORT=12354
export CUDA_VISIBLE_DEVICES="1"
export NCCL_P2P_DISABLE=1

python supervised_pretrain.py \
    --batch_size 512 \
    --epochs 10 \
    --save_path "./log-pretrain/supervised-pretrain/checkpoints_mamba2_tuab" \
    --tuab_root \
    --tuev_root  \
    --chb_mit_root "/mnt/replace_disk/EEG_data/CHB-MIT/clean_segments"\
    --crowd_source_root \
    --n_channels 16 \
    --lr 1e-5 \
    --step_wise 1000 \
    --weight_decay 1e-4 \
    --pretrained_model_path  "./log-pretrain/unsupervised/tuall_checkpoints_mamba2_freqmix/epoch=99_step=153500.ckpt"
    