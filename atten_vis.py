import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional
from torchvision import transforms
from parser_cifar import get_args
import torchvision
# import pandas as pd
import csv

args = get_args()
from model_for_cifar.deit import  deit_small_patch16_224
model_a = deit_small_patch16_224(pretrained=True, img_size=32, num_classes=10, patch_size=4, args=args).cuda()
model_a = nn.DataParallel(model_a)
model_a.load_state_dict(torch.load('./results/deit_small_at_bgat-features/checkpoint_40')['state_dict'])

model_b = deit_small_patch16_224(pretrained=True, img_size=32, num_classes=10, patch_size=4, args=args).cuda()
model_b = nn.DataParallel(model_b)
model_b.load_state_dict(torch.load('./results/deit_small_at/checkpoint_40')['state_dict'])

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2023, 0.1994, 0.2010)
# 画像の読み込みと前処理 (例)
transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
# input_image = testset[0][0].unsqueeze(0).cuda()
input_image, label = iter(testloader).next()
input_image = input_image.cuda()
# breakpoint()

# フォワードパス
model_a.eval()
model_b.eval()
with torch.no_grad():
    model_a.module.forward_features(input_image, return_attention=True)
    model_b.module.forward_features(input_image, return_attention=True)
    attention_maps_a = model_a.module.attention_maps
    attention_maps_b = model_b.module.attention_maps
    print(attention_maps_a[0].shape)
    print('len : ', len(attention_maps_a))



for i in range(12):
    for j in range(6):
        attn_a = attention_maps_a[i][0, j]  # i-th Block, batch num, j-th head
        attn_b = attention_maps_b[i][:, j]
        A_f = attn_a.view(attn_a.shape[0], -1)
        B_f = attn_b.view(attn_b.shape[0], -1)
        
        # cosine_sim = torch.nn.functional.cosine_similarity(A_f, B_f, dim=1)
        
        attention = attn_a.detach().cpu().numpy()  # NumPy配列に変換
        

        # ヒートマップとして表示
        plt.imshow(attention, cmap='viridis')
        plt.colorbar()
        plt.title(f"Attention Map from {j} Head")
        plt.show()
        plt.savefig(f'./atten_vis/robust-2/attention_map_layer{i}_head{j}.png')
        plt.close()

# with open('at-at_cos_sim_results.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['layer', 'head', 'cosine_similarity'])
#     cos_sim_lst = []
#     for i in range(12):
#         for j in range(6):
#             attn_a = attention_maps_a[i][:, j]
#             attn_b = attention_maps_b[i][:, j]
#             A_f = attn_a.view(attn_a.shape[0], -1)
#             B_f = attn_b.view(attn_b.shape[0], -1)
#             cosine_sim = torch.nn.functional.cosine_similarity(A_f, B_f, dim=1)
#             writer.writerow([i, j, cosine_sim.mean().item()])
#             cos_sim_lst.append(cosine_sim.mean().item())
            
#     writer.writerow(['mean', 'mean', sum(cos_sim_lst) / len(cos_sim_lst)])
            