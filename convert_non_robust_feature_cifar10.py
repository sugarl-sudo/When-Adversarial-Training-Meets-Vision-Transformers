from notebook.models.wideresnet import WideResNetProp
from models.resnet import ResNet50
import torch
import torch.nn as nn
import torchvision
import os
from torchvision import transforms
from PIL import Image
import random
from tqdm import tqdm
from torch.autograd import Variable
from parser_cifar import get_args


data_dir = 'data/'
out_dir = 'data/cifar10/non_robust_features-vit'
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def l2_pgd(x_natural, x_random, y, model, epsilon=0.5, perturb_steps=100, step_size=0.1):
    batch_size = len(x_natural)
    delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
    delta = Variable(delta.data, requires_grad=True)
    optimizer = torch.optim.SGD([delta], lr=step_size)
    # t = (y.clone().detach().cuda() + 1) % 10
    t = torch.randint(0, 10, (batch_size,)).cuda()
    # breakpoint()
    criterion = nn.CrossEntropyLoss()
    for i in tqdm(range(perturb_steps), desc='PGD Iterations'):
        adv = x_natural + delta
        # optmize
        optimizer.zero_grad()
        with torch.enable_grad():
            loss = criterion(model(adv), t)
        
        # if i % 100 == 0:
        #     print(f'Loss at iteration {i}: {loss.item()}')
        loss.backward()
        grad_norm = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
        delta.grad.div_(grad_norm.view(-1, 1, 1, 1))
        # avoid nan or inf if gradient is 0
        if (grad_norm == 0).any():
            delta.grad[grad_norm == 0] = torch.randn_like(delta.grad[grad_norm == 0])
        optimizer.step()
        # projectoin
        delta.data.add_(x_natural.data)
        delta.data.clamp_(0, 1).sub_(x_natural.data)
        delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        # delta.data.clamp_(0, 1)
    
    x_adv = Variable(x_natural + delta, requires_grad=False)
    # breakpoint()
    return x_adv, t


def get_random_batch(dataset, batch_size):
    # random sampling from dataset
    indices = random.sample(range(len(dataset)), batch_size)
    random_samples = [dataset[i][0] for i in indices]
    return torch.stack(random_samples)  # stack into a batch

def save_adv_examples(x_adv, y, out_dir, batch_num):
    # Ensure output directories exist
    for class_name in cifar10_classes:
        label_dir = os.path.join(out_dir, class_name)
        os.makedirs(label_dir, exist_ok=True)
    
    # Save each adversarial example in the corresponding label directory
    for idx, (adv_img, label) in enumerate(zip(x_adv, y)):
        class_name = cifar10_classes[label.item()]
        label_dir = os.path.join(out_dir, class_name)
        file_name = f"adv_img_batch{batch_num}_idx{idx}.png"
        file_path = os.path.join(label_dir, file_name)
        # Convert to PIL image and save
        adv_img = adv_img.detach().cpu()  # move to CPU
        adv_img = (adv_img * 255).clamp(0, 255).byte()  # scale to 0-255 and convert to byte
        adv_img = transforms.ToPILImage()(adv_img)  # convert to PIL image
        adv_img.save(file_path)
        


def main():
    # Load Robust model
    # model = WideResNetProp(depth=34)
    # model = ResNet50().cuda()
    # model.load_state_dict(torch.load('../model-wideres-epoch99.pt'))
    # model.load_state_dict(torch.load('./results/model-cifar-ResNet50/model-res-epoch100.pt'))
    from model_for_cifar.deit import deit_small_patch16_224
    from parser_cifar import get_args
    args_vit = get_args()
    model = deit_small_patch16_224(pretrained=True, num_classes=10, img_size=args_vit.crop, patch_size=args_vit.patch, args=args_vit).cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load('./results/vit-clean-standard/model-deit-epoch40.pt'))
    
    model = model.cuda()
    trans = transforms.Compose([
        transforms.ToTensor(),
    ])
    # Load CIFAR10 dataset
    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False, transform=trans)
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4)
    for batch_num, (x_natural, y) in enumerate(tqdm(data_loader, desc='Processing Batches')):
        x_natural = x_natural.cuda()
        x_random = get_random_batch(train_dataset, len(x_natural))
        x_random = x_random.cuda()
        y = y.cuda()
        x_adv, t = l2_pgd(x_natural, x_random, y, model, epsilon=1.5, perturb_steps=100, step_size=0.1)
        
        save_adv_examples(x_adv, t, out_dir, batch_num)


if __name__ == '__main__':
    main()