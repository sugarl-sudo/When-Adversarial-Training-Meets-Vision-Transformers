# AT
# CUDA_VISIBLE_DEVICES=0 python train_cifar.py --model "deit_tiny_patch16_224" --out-dir "./pgd_vanilla" --method 'AT' --seed 0
# TRADES
# CUDA_VISIBLE_DEVICES=1 python train_cifar.py --model "deit_tiny_patch16_224" --out-dir "./trades_vanilla" --method 'TRADES' --seed 0 
# MART
CUDA_VISIBLE_DEVICES=2 python train_cifar.py --model "deit_tiny_patch16_224" --out-dir "./results/mart_vanilla" --method 'MART' --seed 0 > mart_vanilla.log 2>&1 & 
# TRADES + deit_small 
CUDA_VISIBLE_DEVICES=3 python train_cifar.py --model "deit_small_patch16_224" --out-dir "./results/trades_vanilla" --method 'TRADES' --seed 0 > trades_vanilla.log 2>&1 & 