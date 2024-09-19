# CUDA_VISIBLE_DEVICES=0 python train_cifar.py --model "deit_tiny_patch16_224" --out-dir "./pgd_vanilla" --method 'AT' --seed 0  &
# CUDA_VISIBLE_DEVICES=1 python train_cifar.py --model "deit_tiny_patch16_224" --n_w 10 --out-dir "./pgd_architecture" --method 'AT' --seed 0 --ARD --PRM &
# CUDA_VISIBLE_DEVICES=2 python train_cifar.py --model "deit_tiny_patch16_224" --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  &
# CUDA_VISIBLE_DEVICES=3 python train_cifar.py --model "deit_tiny_patch16_224" --n_w 10 --out-dir "./trades_architecture" --method 'TRADES' --seed 0 --ARD --PRM &
# CUDA_VISIBLE_DEVICES=4 python train_cifar.py --model "deit_tiny_patch16_224" --out-dir "./mart_vanilla" --method 'MART' --seed 0  &
# CUDA_VISIBLE_DEVICES=5 python train_cifar.py --model "deit_tiny_patch16_224" --n_w 10 --out-dir "./mart_architecture" --method 'MART' --seed 0 --ARD --PRM &
# wait

# # non-robust feature dataset
# out_dir="./results/deit_small_non_robust_shift"
# mkdir -p $out_dir
# # deit small train robust feature dataset
# CUDA_VISIBLE_DEVICES=1 python train_clean_vit.py --epochs 40 \
#             --weight-decay 1e-4 \
#             --momentum 0.9 \
#             --model-dir $out_dir \
#             --dataset-path './data/cifar10/non_robust_features-vit-shift' \
#             --batch-size 128 > $out_dir/process.log

# # wideresnet train non-robust feature dataset
# out_dir="./results/wideres-clean-non_robust_feature_dataset"
# mkdir -p $out_dir
# CUDA_VISIBLE_DEVICES=2 python train_clean_cifar10.py --epochs 100 \
#             --weight-decay 2e-4 \
#             --momentum 0.9 \
#             --model-dir $out_dir \
#             --batch-size 128 > $out_dir/process.log


# # vit-small train AT
# out_dir="./results/deit_small_at"
# mkdir -p $out_dir
# CUDA_VISIBLE_DEVICES=2 python train_cifar.py --model "deit_small_patch16_224" \
#             --method "AT" \
#             --out-dir $out_dir \
#             --seed 0 &

# vit-small train AT with LBGAT
out_dir="./results/deit_small_at_cos_attn-6"
mkdir -p $out_dir
CUDA_VISIBLE_DEVICES=1s python train_cifar.py --model "deit_small_patch16_224" \
            --method "AT" \
            --out-dir $out_dir \
            --seed 0 \
            --lbgat \
            --lbgat-beta 0.1 \
            --mse-rate-attn 10.0 \
            --mse-rate-feat 10.0 \
            --features \
            --teacher-model-path "./results/deit_small_standard/model-deit-epoch40.pt" &

# # vit-small train AT with LBGAT
# out_dir="./results/deit_small_at_lbgat-features"
# mkdir -p $out_dir
# CUDA_VISIBLE_DEVICES=1 python train_cifar.py --model "deit_small_patch16_224" \
#             --method "AT" \
#             --out-dir $out_dir \
#             --seed 0 \
#             --lbgat \
#             --lbgat-beta 1.0 \
#             --mse-rate 1.0 \
#             --features \
#             --teacher-model-path "./results/deit_small_standard/model-deit-epoch40.pt" &



# # convit-small train TRADES
# out_dir="./results/convit_small_trades"
# mkdir -p $out_dir
# CUDA_VISIBLE_DEVICES=2 python train_cifar.py --model "convit_small" \
#             --method "TRADES" \
#             --seed 0 > $out_dir/process.log &

# # vit-small train standard
# out_dir="./results/deit_small_standard"
# mkdir -p $out_dir
# CUDA_VISIBLE_DEVICES=1 python train_clean_vit.py --epoch 40 \
#             --weight-decay 1e-4 \
#             --model-dir $out_dir \
#             --momentum 0.9 > $out_dir/process.log &
