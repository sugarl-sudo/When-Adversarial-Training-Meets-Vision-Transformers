# When-Adversarial-Training-Meets-Vision-Transformers
Official implementation of "When Adversarial Training Meets Vision Transformers: Recipes from Training to Architecture" published at NeurIPS 2022. 
## Requirements
Run `pip install requirement.txt` to install all requrements!
## CIFAR-10
### Vanilla adversarial defense methods：
```python
# AT
CUDA_VISIBLE_DEVICES=0 python train_cifar.py --model "deit_tiny_patch16_224" --out-dir "./pgd_vanilla" --method 'AT' --seed 0
# TRADES
CUDA_VISIBLE_DEVICES=1 python train_cifar.py --model "deit_tiny_patch16_224" --out-dir "./trades_vanilla" --method 'TRADES' --seed 0
# MART
CUDA_VISIBLE_DEVICES=2 python train_cifar.py --model "deit_tiny_patch16_224" --out-dir "./mart_vanilla" --method 'MART' --seed 0
```
You can use `--model` to select other ViT variants to train.
### Example for AT after combining ARD and PRM：
```python
CUDA_VISIBLE_DEVICES=0 python train_cifar.py --model "deit_tiny_patch16_224" --n_w 10 --out-dir "./pgd_architecture" --method 'AT' --seed 0 --ARD --PRM
```
You can use `--method` to select other defense methods!




## Imagenette
First, you need to download the ImageNette-v1 dataset (the old version of ImageNette) to the local path `./data` from [here](https://s3.amazonaws.com/fast-ai-imageclas/imagenette.tgz).
### Vanilla adversarial defense methods：
```python
# AT
CUDA_VISIBLE_DEVICES=0 python train_imagenette.py --model "deit_tiny_patch16_224" --out-dir "./pgd_vanilla" --method 'AT' --seed 0
# TRADES
CUDA_VISIBLE_DEVICES=1 python train_imagenette.py --model "deit_tiny_patch16_224" --out-dir "./trades_vanilla" --method 'TRADES' --seed 0
# MART
CUDA_VISIBLE_DEVICES=2 python train_imagenette.py --model "deit_tiny_patch16_224" --out-dir "./mart_vanilla" --method 'MART' --seed 0
```
You can use `--model` to select other ViT variants to train.
### Example for AT after combining ARD and PRM：
```python
CUDA_VISIBLE_DEVICES=0 python train_imagenette.py --model "deit_tiny_patch16_224" --n_w 10 --out-dir "./pgd_architecture" --method 'AT' --seed 0 --ARD --PRM
```
You can use `--method` to select other defense methods!

## ImageNet-1k
### Vanilla AT：
```python
python train_imagenet.py --model "swin_base_patch4_window7_224_in22k" --out-dir "./pgd_vanilla" --seed 0
```
You can use `--model` to select other ViT variants to train.
### Example for AT after combining ARD and PRM：
```python
python train_imagenet.py --model "swin_base_patch4_window7_224_in22k" --n_w 2 --out-dir "./pgd_architecture" --seed 0 --ARD --PRM
```




## Acknowlegements
This repository is built upon the following four repositories:<br/>
https://github.com/yaodongyu/TRADES
<br/>
https://github.com/YisenWang/MART
<br/>
https://github.com/rwightman/pytorch-image-models
<br/>
https://github.com/RulinShao/on-the-adversarial-robustness-of-visual-transformer.



## Cite this
If you find our code is useful, we sincerely hope you could cite our accompanying paper!
```
@article{mo2022adversarial,
  title={When Adversarial Training Meets Vision Transformers: Recipes from Training to Architecture},
  author={Mo, Yichuan and Wu, Dongxian and Wang, Yifei and Guo, Yiwen and Wang, Yisen},
  journal={arXiv preprint arXiv:2210.07540},
  year={2022}
}
```
