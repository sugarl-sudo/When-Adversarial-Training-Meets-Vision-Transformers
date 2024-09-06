# CUDA_VISIBLE_DEVICES=0 python train_cifar.py --model "deit_tiny_patch16_224" --out-dir "./pgd_vanilla" --method 'AT' --seed 0  &
# CUDA_VISIBLE_DEVICES=1 python train_cifar.py --model "deit_tiny_patch16_224" --n_w 10 --out-dir "./pgd_architecture" --method 'AT' --seed 0 --ARD --PRM &
# CUDA_VISIBLE_DEVICES=2 python train_cifar.py --model "deit_tiny_patch16_224" --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  &
# CUDA_VISIBLE_DEVICES=3 python train_cifar.py --model "deit_tiny_patch16_224" --n_w 10 --out-dir "./trades_architecture" --method 'TRADES' --seed 0 --ARD --PRM &
# CUDA_VISIBLE_DEVICES=4 python train_cifar.py --model "deit_tiny_patch16_224" --out-dir "./mart_vanilla" --method 'MART' --seed 0  &
# CUDA_VISIBLE_DEVICES=5 python train_cifar.py --model "deit_tiny_patch16_224" --n_w 10 --out-dir "./mart_architecture" --method 'MART' --seed 0 --ARD --PRM &
# wait

# # deit small train robust feature dataset
# CUDA_VISIBLE_DEVICES=3 python train_clean_cifar10.py --epochs 40 \
#             --weight-decay 1e-4 \
#             --momentum 0.9 \
#             --batch-size 128 > ./results/vit-clean-robust_feature_dataset/process.log


# wideresnet train non-robust feature dataset
out_dir="./results/wideres-clean-non_robust_feature_dataset"
mkdir -p $out_dir
CUDA_VISIBLE_DEVICES=2 python train_clean_cifar10.py --epochs 100 \
            --weight-decay 2e-4 \
            --momentum 0.9 \
            --model-dir $out_dir \
            --batch-size 128 > $out_dir/process.log


# # vit train standard
# out_dir="./results/vit-clean-standard"
# mkdir -p $out_dir
# CUDA_VISIBLE_DEVICES=1 python train_clean_cifar10.py --epochs 40 \
#             --weight-decay 1e-4 \
#             --momentum 0.9 \
#             --model-dir $out_dir \
#             --batch-size 128 > $out_dir/process.log