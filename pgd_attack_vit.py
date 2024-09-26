from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from models.wideresnet import *
from models.resnet import *
from tqdm import tqdm
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD Attack Evaluation')
parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=2.,
                    help='perturbation')
parser.add_argument('--num-steps', default=20,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.5,
                    help='perturb step size')
parser.add_argument('--random',
                    default=True,
                    help='random initialization for PGD')
parser.add_argument('--model-path',
                   default='./results/model-cifar-wideResNet34-10-clean-robust_feature_dataset/model-wideres-epoch90.pt',
                   help='model for white-box attack evaluation')
# parser.add_argument('--model-path',
#                     default='./results/vit-clean-robust_feature_dataset/model-deit-epoch40.pt',
#                     help='model for white-box attack evaluation')
parser.add_argument('--source-model-path',
                    default='./results/model-cifar-wideResNet34-10-clean-robust_feature_dataset/model-wideres-epoch90.pt',
                    help='source model for black-box attack evaluation')
parser.add_argument('--target-model-path',
                    default='./results/model-cifar-wideResNet34-10-clean-robust_feature_dataset/model-wideres-epoch90.pt',
                    help='target model for black-box attack evaluation')
parser.add_argument('--white-box-attack', default=True,
                    help='whether perform white-box attack')

args = parser.parse_args()

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
inv_cifar10_mean = (-0.4914, -0.4822, -0.4465)
inv_cifar10_std = (1/0.2471, 1/0.2435, 1/0.2616)
# set up data loader
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std),
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

seed = 0  # 固定したいシード値
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# random.seed(seed)
# np.random.seed(seed)

def _pgd_img_save(imgs, path):
    # unnomalize = transforms.Normalize(mean=inv_cifar10_mean, std=inv_cifar10_std)
    unnomalize = transforms.Normalize(
        mean=[-m/s for m, s in zip(cifar10_mean, cifar10_std)],
        std=[1/s for s in cifar10_std]
    )
    imgs = unnomalize(imgs)
    imgs = torch.clamp(imgs, 0, 1)
    imgs = imgs * 255
    grid = torchvision.utils.make_grid(imgs.cpu().detach(), nrow=8, padding=2)
    plt.figure(figsize=(10, 5))
    plt.imshow(grid.permute(1, 2, 0).int().numpy())
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight')
    # breakpoint()
    plt.close()

def _pgd_whitebox(model,
                  X,
                  y,
                  batch_num,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):
    mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
    std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()
    upper_limit = ((1 - mu) / std)
    lower_limit = ((0 - mu) / std)
    epsilon = (epsilon / 255.) / std
    alpha = (step_size / 255.) / std
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    if args.random:
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = torch.clamp(delta, lower_limit - X, upper_limit - X)
        delta = Variable(delta, requires_grad=True)

    # l-inf PGD
    for _ in range(num_steps):
        output = model(X + delta)
        index = torch.where(output.max(1)[1] == y)
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        grad = delta.grad.detach()
        d = delta[index[0], :, :, :]
        g = grad[index[0], :, :, :]
        d = torch.clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
        d = torch.clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
        delta.data[index[0], :, :, :] = d
        delta.grad.zero_()
    X_pgd = Variable(X + delta, requires_grad=False)
    # l-2 PGD
    # for _ in range(num_steps):
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    delta = delta[index[0]]
    X = X[index[0]]
    # breakpoint()
    # print('err pgd (white-box): ', err_pgd)
    # delta_path = args.model_path.replace('checkpoint_40', f'pgd-standard/pgd_delta_batch{batch_num}.png')
    # X_path = args.model_path.replace('checkpoint_40', f'pgd-standard/pgd_X_batch{batch_num}.png')
    # X_adv_path = args.model_path.replace('checkpoint_40', f'pgd-standard/pgd_X_adv_batch{batch_num}.png')
    # _pgd_img_save(delta[:24], delta_path)
    # _pgd_img_save(X[:24], X_path)
    # _pgd_img_save(X_pgd[:24], X_adv_path)
    # breakpoint()
    return err, err_pgd


def _pgd_blackbox(model_target,
                  model_source,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):
    out = model_target(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model_source(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    err_pgd = (model_target(X_pgd).data.max(1)[1] != y.data).float().sum()
    # print('err pgd black-box: ', err_pgd)
    return err, err_pgd


def eval_adv_test_whitebox(model, device, test_loader):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    natural_err_total = 0

    for i, (data, target) in enumerate(tqdm(test_loader)):
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_whitebox(model, X, y, i)
        robust_err_total += err_robust
        natural_err_total += err_natural
    print('natural_err_total: ', natural_err_total)
    print('natural accuracy: ', 1 - natural_err_total / len(test_loader.dataset))
    print('robust_err_total: ', robust_err_total)
    print('robust accuracy: ', 1 - robust_err_total / len(test_loader.dataset))


def eval_adv_test_blackbox(model_target, model_source, device, test_loader):
    """
    evaluate model by black-box attack
    """
    model_target.eval()
    model_source.eval()
    robust_err_total = 0
    natural_err_total = 0

    for data, target in tqdm(test_loader):
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_blackbox(model_target, model_source, X, y)
        robust_err_total += err_robust
        natural_err_total += err_natural
    print('natural_err_total: ', natural_err_total)
    print('natural accuracy: ', 1 - natural_err_total / len(test_loader.dataset))
    print('robust_err_total: ', robust_err_total)
    print('robust accuracy: ', 1 - robust_err_total / len(test_loader.dataset))
    


def main():

    if args.white_box_attack:
        # white-box attack
        print('pgd white-box attack')
        
        # Vit
        from model_for_cifar.deit import deit_small_patch16_224, deit_tiny_patch16_224
        from model_for_cifar.vit import vit_base_patch16_224, vit_small_patch16_224
        from parser_cifar import get_args
        args_vit = get_args()
        # args_vit.model = 'deit_small_patch16_224'
        # args_vit.model = 'vit_small_patch16_224'
        model = deit_small_patch16_224(pretrained=True, num_classes=10, img_size=32, patch_size=4, args=args_vit).cuda()
        # model = vit_base_patch16_224(pretrained=True, num_classes=10, img_size=32, patch_size=4, args=args_vit).cuda()
        # model = deit_tiny_patch16_224(pretrained=True, num_classes=10, img_size=32, patch_size=4, args=args_vit).cuda()
        # model = vit_small_patch16_224(pretrained=True, num_classes=10, img_size=32, patch_size=4, args=args_vit).cuda()
        # breakpoint()
        model = nn.DataParallel(model)
        model.eval()
        print('model-path', args.model_path)
        # breakpoint()
        # import pickle
        # model.load_state_dict(torch.load(args.model_path, encoding='latin1', map_location='cuda:0', pickle_module=pickle))
        model.load_state_dict(torch.load(args.model_path)['state_dict'])
        print(model)
        print(args_vit)
        
        # # wideresnet
        # model = WideResNet(depth=34).to(device)
        # model.load_state_dict(torch.load(args.model_path))

        eval_adv_test_whitebox(model, device, test_loader)
    else:
        # black-box attack
        print('pgd black-box attack')
        model_target = WideResNet(depth=34).to(device)
        model_target.load_state_dict(torch.load(args.target_model_path))
        model_source = WideResNet(depth=34).to(device)
        model_source.load_state_dict(torch.load(args.source_model_path))

        eval_adv_test_blackbox(model_target, model_source, device, test_loader)


if __name__ == '__main__':
    main()
