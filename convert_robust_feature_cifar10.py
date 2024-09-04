from notebook.models.widresnet import WideResNetProp
import torch
import torch.nn as nn


data_dir = 'data/'
out_dir = 'data/cifar10/robust_features'

def l2_pgd(x_natural, x_random, y, model, epsilon=0.1, perturb_steps=1000):
    batch_size = len(x_natural)
    x_adv = x_random.clone().detach().requires_grad_(True)
    optmizer = torch.optim.SGD([x_adv], lr=espilon)
    criterion = nn.MSELoss()
    for _ in range(perturb_steps):
        # optmize
        optmizer.zero_grad()
        with torch.enable_grad():
            loss = criterion(model(x_adv)[1], model(x_natural)[1])
            
        loss.backward()
        grad_norm = x_adv.grad.view(batch_size, -1).norm(p=2, dim=1)
        x_adv.grad.div_(grad_norm.view(-1, 1, 1, 1))
        # avoid nan or inf if gradient is 0
        if (grad_norm == 0).any():
            x_adv.grad[grad_norm == 0] = torch.randn_like(x_adv.grad[grad_norm == 0])
        optimizer.step()
        
    return x_adv, y

def main():
    # Load Robust model
    model = WideResNetProp(depth=34)
    model.load_state_dict(torch.load('../model-wideres-epoch99.pt'))
    transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    # Load CIFAR10 dataset
    train_dataset = torchvison.datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transforms),
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    for batch_num, (x_natural, y) in enumerate(data_loader):
        x_natural = x_natural.cuda()
        y = y.cuda()
        x_adv, _ = l2_pgd(x_natural, y, model)
    
    
if __main__ == '__name__':
    main()