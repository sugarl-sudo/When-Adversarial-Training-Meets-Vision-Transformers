import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from parser_cifar import get_args
import torchvision

args = get_args()
from model_for_cifar.deit import  deit_small_patch16_224
model = deit_small_patch16_224(pretrained=True, img_size=32, num_classes=10, patch_size=4, args=args).cuda()
model = nn.DataParallel(model)
model.load_state_dict(torch.load('./results/deit_small_standard/model-deit-epoch40.pt'))

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2023, 0.1994, 0.2010)
# 画像の読み込みと前処理 (例)
transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
input_image = testset[0][0].unsqueeze(0).cuda()

# フォワードパス
model.eval()
with torch.no_grad():
    model.module.forward_features(input_image, return_attention=True)
    attention_maps = model.module.attention_maps
    print(attention_maps[0].shape)
    print('len : ', len(attention_maps))

# Attention Mapsの可視化
# 例えば、最初のAttention Mapの最初のヘッドの可視化
if attention_maps:
    for i in range(12):
        for j in range(6):
            attention = attention_maps[i][0][j]  # 最初のBlock, 最初のバッチ, 最初のヘッドのAttention Map
            attention = attention.detach().cpu().numpy()  # NumPy配列に変換

            # ヒートマップとして表示
            plt.imshow(attention, cmap='viridis')
            plt.colorbar()
            plt.title(f"Attention Map from {j} Head")
            plt.show()
            plt.savefig(f'./atten_vis/standard/attention_map_layer{i}_head{j}.png')
            plt.close()
        # plt.clear()
else:
    print("No attention maps were captured.")
