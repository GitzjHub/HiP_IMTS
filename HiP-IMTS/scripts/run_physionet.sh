### tPatchGNN ###
patience=10
gpu=0

for seed in {1..5}
do
    python run_models.py \
    --dataset physionet --history 12 \
    --patience $patience --batch_size 32 --lr 1e-3 \
    --patch_sizes [2,4] --nhead 1 --elayer 1 --nlayer 1 \
    --te_dim 10 --node_dim 10 --hid_dim 64 \
    --outlayer Linear --seed $seed --gpu $gpu
done
