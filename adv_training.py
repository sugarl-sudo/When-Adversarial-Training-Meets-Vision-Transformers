import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
device = torch.device("cuda")
from utils import * #EWC
from non_local_embedded_gaussian import NONLocalBlock2D

import ot
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def clean_loss(model,
                X,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):

    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(X)
    #
    loss_natural = F.cross_entropy(logits, y)
    loss = loss_natural
    return loss



def adv_loss(model,
                X,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):
    model.eval()
    batch_size = len(X)
    # generate adversarial example
    # x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    X = X.detach() + 0.001 * torch.randn(X.shape).cuda().detach()
    out = model(X)
    X_pgd = Variable(X.data, requires_grad=True)
    #
    #if distance == 'l_inf':
    #    for _ in range(perturb_steps):
    #        x_adv.requires_grad_()
    #        with torch.enable_grad():
    #            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
    #                                   F.softmax(model(x_natural), dim=1))
    #        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
    #        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
    #        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
    #        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    #elif distance == 'l_2':
    #    delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
    #    delta = Variable(delta.data, requires_grad=True)

    #    # Setup optimizers
    #    optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

    #    for _ in range(perturb_steps):
    #        adv = x_natural + delta

    #        # optimize
    #        optimizer_delta.zero_grad()
    #        with torch.enable_grad():
    #            loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
    #                                       F.softmax(model(x_natural), dim=1))
    #        loss.backward()
    #        # renorming gradient
    #        grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
    #        delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
    #        # avoid nan or inf if gradient is 0
    #        if (grad_norms == 0).any():
    #            delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
    #        optimizer_delta.step()

    #        # projection
    #        delta.data.add_(x_natural)
    #        delta.data.clamp_(0, 1).sub_(x_natural)
    #        delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
    #    x_adv = Variable(x_natural + delta, requires_grad=False)
    #else:
    #    x_adv = torch.clamp(x_adv, 0.0, 1.0)i

    for _ in range(perturb_steps):
        #opt = optim.SGD([X_pgd], lr=1e-3)
        #opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    model.train()

    x_adv = X_pgd
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(X)
    logits_adv = model(x_adv)
    #
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = F.cross_entropy(logits_adv, y)
    loss = 1./2 *loss_natural + 1./2 * loss_robust
    #loss = loss_robust
    return loss


def prop_loss(model,
              model_teacher,
                X,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):
    model.eval()
    model_teacher.eval()
    batch_size = len(X)
    # generate adversarial example
    # x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    #out, _ = model(X)
    X_pgd = Variable(X.data, requires_grad=True)

    for _ in range(perturb_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()
    
        with torch.enable_grad():
            out_adv, _ = model(X_pgd)
            loss = nn.CrossEntropyLoss()(out_adv, y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    model.train()

    x_adv = X_pgd
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits, features = model(X)
    logits_adv, features_adv = model(x_adv)
    ##
    _,  features_teach = model_teacher(X)
    #_,  features_adv_teach = model_teacher(x_adv)
    #
    #loss_natural = F.cross_entropy(logits, y)
    loss_robust = F.cross_entropy(logits_adv, y)
    #loss_inter = 1./2 * F.mse_loss(features, features_teach) +\
    #             1./2 * F.mse_loss(features_adv, features_teach)
    #loss_inter = F.mse_loss(features_adv, features_teach)
    loss_inter = (1 - F.cosine_similarity(features_adv, features_teach).abs()).mean()
    ##
    #labels = Variable((features_teach > 0).float())
    #inter = nn.Sigmoid()(features_adv*100)
    #
    #loss_inter = F.mse_loss(inter, labels) 
    ##
    #loss = 1./2 * loss_natural + 1./2 * loss_robust + 100*loss_inter
    loss = loss_robust + 5 * loss_inter
    return loss



def prop_loss_da(model,
                 #model_teacher,
                 #model_teacher_adv,
                 X,
                 y,
                 optimizer,
                 step_size=0.003,
                 epsilon=0.031,
                 perturb_steps=10,
                 beta=1.0,
                 distance='l_inf'):
    model.eval()
    batch_size = len(X)
    # generate adversarial example
    # x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    #out, _ = model(X)
    X_pgd = Variable(X.data, requires_grad=True)

    for _ in range(perturb_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            out_adv, _ = model(X_pgd)
            loss = nn.CrossEntropyLoss()(out_adv, y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    model.train()

    x_adv = X_pgd
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits, logits_da = model(X)
    logits_adv, logits_da_adv = model(x_adv)
    ##
    #
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = F.cross_entropy(logits_adv, y)

    da_label_clean = y*0
    da_label_adv = y*0+1

    loss_da =  F.cross_entropy(logits_da, da_label_clean)
    loss_da_robust = F.cross_entropy(logits_da_adv, da_label_adv)

    #labels = Variable((features_teach > 0).float())
    #labels_adv = Variable((features_adv_teach > 0).float())
    #inter = nn.Sigmoid()(features*100)
    #inter_adv = nn.Sigmoid()(features_adv*100)
    #
    #loss_inter = 1./2 * F.mse_loss(inter, labels) +\
    #             1./2 * F.mse_loss(inter_adv, labels)

    loss = 1./2 * loss_natural + 1./2 * loss_robust
    loss_domain = 1/2. * loss_da + 1/2. * loss_da_robust
    return loss, 2*loss_domain


def prop_triplet_loss(model,
                      model_teacher,
                      X_tri,
                      y,
                      optimizer,
                      epoch,
                      step_size=0.003,
                      epsilon=0.031,
                      perturb_steps=10,
                      beta=1.0,
                      distance='l_inf'):
    model.eval()
    model_teacher.eval()
    X = X_tri[0]
    batch_size = len(X)
    alpha = 0.03
    # generate adversarial example
    # x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    #out, _ = model(X)
    X_pgd = Variable(X.data, requires_grad=True)

    for _ in range(perturb_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            out_adv, _ = model(X_pgd)
            loss = nn.CrossEntropyLoss()(out_adv, y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    model.train()

    x_adv = X_pgd
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    #logits, features = model(X)
    logits_adv, features_adv = model(x_adv)
    ##
    logits,  features = model(X)
    logits_teach,  features_teach = model_teacher(X)
    _,  features_pos = model(X_tri[1])
    _,  features_neg = model(X_tri[2])
    #
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = F.cross_entropy(logits_adv, y)
    #loss_inter = 1./2 * F.mse_loss(features, features_teach) +\
    #             1./2 * F.mse_loss(features_adv, features_teach)
    #loss_inter = F.mse_loss(features_adv, features_teach)
    loss_inter_1 = (1 - F.cosine_similarity(features, features_teach).abs()).mean()
    loss_inter_2 = (1 - F.cosine_similarity(features_adv, features_teach).abs()).mean()
    ##
    loss = 0.9* loss_natural + 0.1 * loss_robust + 10 * loss_inter_1 + 10 * loss_inter_2
    return loss


def prop_tmc_loss(model,
                  model_teacher,
                  X,
                  y,
                  optimizer,
                  epoch,
                  step_size=0.003,
                  epsilon=0.031,
                  perturb_steps=10,
                  beta=1.0,
                  distance='l_inf'):

    model.eval()
    model_teacher.eval()
    #X = X_tri[0]
    batch_size = len(X)
    m = batch_size
    n = batch_size
    #ewc = EWC(model_teacher, X, y)

    # generate adversarial example
    # x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    #out, _ = model(X)
    X_pgd = Variable(X.data, requires_grad=True)
    #nat_logits, _ = model(X_pgd)
    #MC_targets = MC_labels(nat_logits, y)

    X_n = X + torch.zeros_like(X).uniform_(-epsilon, epsilon)
    X_n = Variable(torch.clamp(X_n, 0, 1.0), requires_grad=True)
    nat_logits, _ = model(X_n)
    MC_targets = MC_labels(nat_logits, y)
    X_pgd = Variable(X_n.data, requires_grad=True)
    #nat_logits, _ = model(X_pgd)

    num_classes = nat_logits.size(1)
    y_gt = one_hot_tensor(y, num_classes, device)
    loss_ce = softCrossEntropy()

    for _ in range(perturb_steps):
        #opt = optim.SGD([X_pgd], lr=1e-3)
        #opt.zero_grad()

        with torch.enable_grad():
            out_adv, _ = model(X_pgd)
            loss = nn.CrossEntropyLoss()(out_adv, MC_targets)
            #loss = ot.sinkhorn_loss_joint_IPOT(1, 0.00, nat_logits,
            #                                   out_adv, None, None,
            #                                   0.01, m, n)

        loss.backward()
        eta = -step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    model.train()

    x_adv = X_pgd
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    #logits, features = model(X)
    logits_adv, features_adv = model(x_adv)
    ##
    logits,  features = model(X)
    logits_teach,  features_teach = model_teacher(X)
    logits_teach_adv,  features_teach_adv = model_teacher(x_adv)
    ##
    denoiser = NONLocalBlock2D(in_channels=features.shape[1], sub_sample=False).to(device)
    features_denoise = denoiser(features_teach)
    #
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = F.cross_entropy(logits_adv, y)
    #loss_inter = 1./2 * F.mse_loss(features, features_teach) +\
    #             1./2 * F.mse_loss(features_adv, features_teach)
    #loss_inter = F.mse_loss(features_adv, features_teach)
    #
    #pos_1 = (1 - F.cosine_similarity(features, features_teach).abs())
    #neg_1 = (1 - F.cosine_similarity(features, features_teach_adv).abs())
    #loss_inter_1 = torch.max(torch.tensor(0., device=device), pos_1 - 0.01*neg_1).mean()
    #
    #pos_2 = (1 - F.cosine_similarity(features_adv, features_teach).abs())
    #neg_2 = (1 - F.cosine_similarity(features_adv, features_teach_adv).abs())
    #loss_inter_2 = torch.max(torch.tensor(0., device=device), pos_2 - 0.5*neg_2).mean()
    #
    loss_inter_1 = (1 - F.cosine_similarity(features, features_denoise).abs()).mean()
    loss_inter_2 = (1 - F.cosine_similarity(features_adv, features).abs()).mean()
    #loss_inter_2 = (1 - F.cosine_similarity(features_adv, features_denoise).abs()).mean()
    ##
    #ls_factor = 0.5
    #y_sm = label_smoothing(y_gt, y_gt.size(1), ls_factor)
    #loss_robust = loss_ce(logits_adv, y_sm.detach())
    ##
    loss = 0*loss_natural + 1*loss_robust + 5 * loss_inter_1 #+ 5 * loss_inter_2
    #loss = loss_robust + 1000000 * ewc.penalty(model) + 5 * loss_inter_2
    #print(ewc.penalty(model))
    return loss



def prop_finetune_loss(model,
                       model_teacher,
                       X,
                       y,
                       optimizer,
                       epoch,
                       #ewc,
                       step_size=0.003,
                       epsilon=0.031,
                       perturb_steps=10,
                       beta=1.0,
                       distance='l_inf'):
    model.eval()
    model_teacher.eval()
    batch_size = len(X)
    # generate adversarial example
    # x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    #out, _ = model(X)
    #X_pgd = X.detach() + 0.001 * torch.randn(X.shape).cuda().detach()
    #X_pgd = Variable(X_pgd.data, requires_grad=True)
    X_pgd = X + torch.zeros_like(X).uniform_(-epsilon, epsilon)
    X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)


    for _ in range(perturb_steps):
        with torch.enable_grad():
            out_adv, _ = model(X_pgd)
            loss = nn.CrossEntropyLoss()(out_adv, y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    model.train()

    x_adv = X_pgd
    ewc = EWC(model_teacher, X, y)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    #logits, features = model(X)
    logits_adv, features_adv = model(x_adv)
    ##
    X_noise = X + torch.zeros_like(X).uniform_(-epsilon, epsilon)
    logits_noise,  features_noise = model(X_noise)
    logits,  features = model(X)
    logits_teach,  features_teach = model_teacher(X)
    logits_teach_adv,  features_teach_adv = model_teacher(x_adv)
    num_classes = logits.size(1)
    y_gt = one_hot_tensor(y, num_classes, device)
    loss_ce = softCrossEntropy()
    #
    loss_natural = F.cross_entropy(logits, y)
    loss_noise = F.cross_entropy(logits_noise, y)
    loss_robust = F.cross_entropy(logits_adv, y)
    #loss_inter = 1./2 * F.mse_loss(features, features_teach) +\
    #             1./2 * F.mse_loss(features_adv, features_teach)
    #loss_inter_2 = F.mse_loss(features_adv, features_teach)
    loss_inter_1 = (1 - F.cosine_similarity(features, features_teach).abs()).mean()
    #loss_inter_2 = (1 - F.cosine_similarity(features_adv, features_teach).abs()).mean()
    loss_inter_2 = (1 - F.cosine_similarity(torch.flatten(features_adv, 1), torch.flatten(features_teach, 1)).abs().mean())
    #loss_inter_1 = F.mse_loss(logits_adv, logits_teach)
    ##
    #loss_kd = nn.KLDivLoss(reduction="batchmean")(torch.log_softmax(logits_adv/2.0, dim=1), 
    #                                              torch.softmax(logits_teach_adv/2.0, dim=1))
    #loss = 0*loss_natural + 1*loss_robust + 5 * loss_inter_1 + 5 * loss_inter_2
    #loss = loss_robust + 5 * epsilon*255./8.  * loss_inter_2
    #ls_factor = 0.3
    #y_sm = label_smoothing(y_gt, y_gt.size(1), ls_factor)
    #loss_robust = loss_ce(logits_adv, y_sm.detach())
    #if epoch <= 10:
    #    loss = 0.5*loss_robust + 0.5*loss_noise#+ 50 * loss_inter_2
    #else:
    #    loss = loss_robust
    loss = loss_robust
    #print(ewc.penalty(model))
    return loss

def prop_ft_learn_loss(model,
                       model_teacher,
                       X,
                       y,
                       optimizer,
                       epoch,
                       #ewc,
                       step_size=0.003,
                       epsilon=0.031,
                       perturb_steps=10,
                       beta=1.0,
                       distance='l_inf'):
    model.eval()
    model_teacher.eval()
    batch_size = len(X)
    #
    # generate adversarial example
    # x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    #out, _ = model(X)
    #X_pgd = X.detach() + 0.001 * torch.randn(X.shape).cuda().detach()
    #X_pgd = Variable(X_pgd.data, requires_grad=True)
    logits,  features = model(X)
    logits_teach,  features_teach = model_teacher(X)
    X_pgd = X + torch.zeros_like(X).uniform_(-epsilon, epsilon)
    X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)


    #for _ in range(perturb_steps):
    #    with torch.enable_grad():
    #        out_adv, features_adv = model(X_pgd)
    #        loss = nn.CrossEntropyLoss()(out_adv, y)
    #    loss.backward()
    #    eta = step_size * X_pgd.grad.data.sign()
    #    X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
    #    eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
    #    X_pgd = Variable(X.data + eta, requires_grad=True)
    #    X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    #model.train()
    #model_teacher.train()

    x_adv = []
    if epoch <= 20:
        thresh = math.cos(math.radians(45))
        K = perturb_steps

        while K>0:
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
            out_adv, features_adv = model(X_pgd)
            pred = out_adv.max(1, keepdim=True)[1]
            iter_index = []

            # Calculate the indexes of adversarial data those still needs to be iterated
            for idx in range(len(pred)):
                dist = F.cosine_similarity(torch.flatten(features_adv[idx], 1), \
                                           torch.flatten(features_teach[idx], 1)).mean()
                if (dist > thresh):
                    iter_index.append(idx)

            # update iter adv
            if len(iter_index) != 0:
                # calculate gradient
                model.zero_grad()
                with torch.enable_grad():
                    loss = nn.CrossEntropyLoss()(out_adv, y)
                    #loss = nn.CrossEntropyLoss()(out_adv, y_cand)
                loss.backward(retain_graph=True)
                grad = X_pgd.grad
                eta = step_size * grad.data[iter_index].sign()
                X_cand = X_pgd.data[iter_index].detach() + eta
                X_cand = torch.min(torch.max(X_cand, X[iter_index].data - epsilon), X[iter_index].data + epsilon)
                X_cand = torch.clamp(X_cand, 0, 1)
                X_pgd = X_pgd.detach() 
                X_pgd[iter_index] = X_cand
                #if K == 1:
                #    print(len(iter_index))

            else:
                #print(K)
                break

            K -= 1
    else:
        for _ in range(perturb_steps):
            with torch.enable_grad():
                out_adv, features_adv = model(X_pgd)
                loss = nn.CrossEntropyLoss()(out_adv, y)
            loss.backward()
            eta = step_size * X_pgd.grad.data.sign()
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
            X_pgd = Variable(X.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    model.train()

    x_adv = X_pgd
    ewc = EWC(model_teacher, X, y)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    #logits, features = model(X)
    logits_adv, features_adv = model(x_adv)
    ##
    #logits,  features = model(X)
    #logits_teach,  features_teach = model_teacher(X)
    logits_teach_adv,  features_teach_adv = model_teacher(x_adv)
    num_classes = logits.size(1)
    y_gt = one_hot_tensor(y, num_classes, device)
    loss_ce = softCrossEntropy()
    #
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = F.cross_entropy(logits_adv, y)
    #loss_inter = 1./2 * F.mse_loss(features, features_teach) +\
    #             1./2 * F.mse_loss(features_adv, features_teach)
    #loss_inter_2 = F.mse_loss(features_adv, features_teach)
    loss_inter_1 = (1 - F.cosine_similarity(features, features_teach).abs()).mean()
    loss_inter_2 = (1 - F.cosine_similarity(features_adv, features_teach).abs()).mean()
    #loss_inter_2 = (1 - F.cosine_similarity(torch.flatten(features_adv, 2), torch.flatten(features_teach, 2)).mean())

    #loss_inter_1 = F.mse_loss(logits_adv, logits_teach)
    ##
    loss = loss_robust #+ 50 * loss_inter_2
    #print(ewc.penalty(model))
    return loss

def cos_sim(v1, v2):
    v1 = v1.detach().cpu()
    v2 = v2.detach().cpu()
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))



def prop_cn_learn_loss(model,
                       model_teacher,
                       X,
                       y,
                       optimizer,
                       epoch,
                       step_size=0.003,
                       epsilon=0.031,
                       perturb_steps=10,
                       beta=1.0,
                       distance='l_inf'):
    model.eval()
    model_teacher.eval()
    batch_size = len(X)
    #
    # generate adversarial example
    # x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    #out, _ = model(X)
    #X_pgd = X.detach() + 0.001 * torch.randn(X.shape).cuda().detach()
    #X_pgd = Variable(X_pgd.data, requires_grad=True)
    logits,  features = model(X)
    logits_teach,  features_teach = model_teacher(X)
    X_pgd = X + torch.zeros_like(X).uniform_(-epsilon, epsilon)
    X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)


    x_adv = []
    if epoch <= 10:
        thresh = math.cos(math.radians(30))
        #thresh = 0.65

        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
        out_adv, features_adv = model(X_pgd)
        pred = out_adv.max(1, keepdim=True)[1]
        iter_index = []

        # Calculate the indexes of adversarial data those still needs to be iterated
        for idx in range(len(pred)):
            dist = cos_sim(torch.flatten(features_adv[idx], 0), \
                           torch.flatten(features_teach[idx], 0)).mean()
            if (dist > thresh):
                iter_index.append(idx)


        print(len(iter_index))
        ##
        #iter_index = random.sample(range(len(pred)), len(iter_index))
        #iter_index = random.sample(range(len(pred)), 64)
        ##
        if len(iter_index) != 0:
            for _ in range(perturb_steps):
                X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
                with torch.enable_grad():
                    out_adv, features_adv = model(X_pgd)
                    loss = nn.CrossEntropyLoss()(out_adv, y)
                loss.backward()
                #grad = X_pgd.grad
                #eta = step_size * grad.data[iter_index].sign()
                eta = step_size * X_pgd.grad.data.sign()
                X_cand = X_pgd.data[iter_index].detach() + eta[iter_index]
                X_cand = torch.min(torch.max(X_cand, X[iter_index].data - epsilon), \
                                                     X[iter_index].data + epsilon)
                X_cand = torch.clamp(X_cand, 0, 1)
                X_pgd = X_pgd.detach()
                #X_pgd = X
                X_pgd[iter_index] = X_cand

    else:
        for _ in range(perturb_steps):
            with torch.enable_grad():
                out_adv, features_adv = model(X_pgd)
                loss = nn.CrossEntropyLoss()(out_adv, y)
            loss.backward()
            eta = step_size * X_pgd.grad.data.sign()
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
            X_pgd = Variable(X.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    model.train()

    x_adv = X_pgd
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits, features = model(X)
    logits_adv, features_adv = model(x_adv)
    ##
    logits_teach,  features_teach = model_teacher(X)
    logits_teach_adv,  features_teach_adv = model_teacher(x_adv)
    #
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = F.cross_entropy(logits_adv, y)
    loss_inter_1 = (1 - F.cosine_similarity(torch.flatten(features, 1), torch.flatten(features_teach, 1)).mean())
    loss_inter_2 = (1 - F.cosine_similarity(features_adv, features_teach).abs()).mean()
    #if len(iter_index) == 0:
    #    loss_inter_2 = 0
    #elif epoch <= 10:
    #    loss_inter_2 = (1 - F.cosine_similarity(torch.flatten(features_adv[iter_index], 1),\
    #                                            torch.flatten(features_teach[iter_index], 1)).mean())
    #else:
    #loss_inter_2 = (1 - F.cosine_similarity(torch.flatten(features_adv, 1), torch.flatten(features_teach, 1)).abs().mean())

    #loss_inter_1 = F.mse_loss(logits_adv, logits_teach)
    ##
    loss_ce = softCrossEntropy()
    loss_kd = torch.nn.MSELoss()(logits_adv, logits_teach)
    loss = loss_robust + 50 * loss_inter_2 #+ 0.5*loss_kd
    return loss
