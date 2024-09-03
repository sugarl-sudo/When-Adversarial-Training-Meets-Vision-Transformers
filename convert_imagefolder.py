import os
import pickle
import numpy as np
from PIL import Image

# CIFAR-10のバッチファイルが保存されているディレクトリ
cifar_dir = 'cifar-10-batches-py'

# 画像を保存するディレクトリ
output_dir = 'cifar-10-images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# CIFAR-10のクラス名
cifar_classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# 各クラスのディレクトリを作成
for class_name in cifar_classes:
    class_dir = os.path.join(output_dir, class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

# CIFAR-10のバッチファイルを読み込む関数
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# データバッチを繰り返し処理して画像を保存
for batch in range(1, 6):
    batch_file = os.path.join(cifar_dir, f'data_batch_{batch}')
    batch_data = unpickle(batch_file)
    images = batch_data[b'data']
    labels = batch_data[b'labels']
    
    for i, (image, label) in enumerate(zip(images, labels)):
        img = image.reshape(3, 32, 32).transpose(1, 2, 0)  # 画像を32x32の形状に変換
        img = Image.fromarray(img)
        class_name = cifar_classes[label]
        img.save(os.path.join(output_dir, class_name, f'{class_name}_{batch}_{i}.png'))

# テストデータの処理（オプション）
test_batch = unpickle(os.path.join(cifar_dir, 'test_batch'))
test_images = test_batch[b'data']
test_labels = test_batch[b'labels']

for i, (image, label) in enumerate(zip(test_images, test_labels)):
    img = image.reshape(3, 32, 32).transpose(1, 2, 0)
    img = Image.fromarray(img)
    class_name = cifar_classes[label]
    img.save(os.path.join(output_dir, class_name, f'{class_name}_test_{i}.png'))
