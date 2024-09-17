import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from parser_cifar import get_args
# VisionTransformerクラスのインスタンス化 (提供されたコードを使用)
# model = VisionTransformer()

args = get_args()
from model_for_cifar.deit import  deit_small_patch16_224
model = deit_small_patch16_224(pretrained=True, img_size=32, num_classes=10, patch_size=4, args=args).cuda()
model = nn.DataParallel(model)

# 画像の読み込みと前処理 (例)
transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ダミーの画像データ (実際には任意の画像を使用)
input_image = torch.randn(1, 3, 32, 32).cuda()  # バッチサイズ1、RGBチャンネル、224x224ピクセル


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
        attention = attention_maps[i][0][0]  # 最初のBlock, 最初のバッチ, 最初のヘッドのAttention Map
        attention = attention.detach().cpu().numpy()  # NumPy配列に変換

        # ヒートマップとして表示
        plt.imshow(attention, cmap='viridis')
        plt.colorbar()
        plt.title("Attention Map from First Head")
        plt.show()
        plt.savefig(f'./atten_vis/attention_map{i}.png')
        plt.close()
        # plt.clear()
else:
    print("No attention maps were captured.")
