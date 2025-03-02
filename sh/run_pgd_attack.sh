out_dir="./results/deit_small_standard"
CUDA_VISIBLE_DEVICES=0 python pgd_attack_vit.py \
    --model-path $out_dir/model-deit-epoch30.pt > $out_dir/pgd_attack.log &

# out_dir="./results/deit_small_at"
# CUDA_VISIBLE_DEVICES=1 python pgd_attack_vit.py \
#     --model-path > $out_dir/checkpoint_40 > $out_dir/pgd_attack.log &

out_dir="./results/deit_small_robust_feature_dataset"
CUDA_VISIBLE_DEVICES=2 python pgd_attack_vit.py \
    --model-path $out_dir/model-deit-epoch30.pt > $out_dir/pgd_attack.log &

out_dir="./results/deit_small_non_robust"
CUDA_VISIBLE_DEVICES=3 python pgd_attack_vit.py \
    --model-path $out_dir/model-deit-epoch30.pt > $out_dir/pgd_attack.log &