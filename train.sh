# AT
CUDA_VISIBLE_DEVICES=0 python train_cifar.py --model "deit_tiny_patch16_224" --out-dir "./pgd_vanilla" --method 'AT' --seed 0
# TRADES
# CUDA_VISIBLE_DEVICES=1 python train_cifar.py --model "deit_tiny_patch16_224" --out-dir "./trades_vanilla" --method 'TRADES' --seed 0
# # MART
# CUDA_VISIBLE_DEVICES=2 python train_cifar.py --model "deit_tiny_patch16_224" --out-dir "./mart_vanilla" --method 'MART' --seed 0