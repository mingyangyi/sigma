from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
import time
import logging
import torchvision
import pickle
import torchvision.transforms as transforms
import numpy as np
import random
import utils
from torch.distributions.normal import Normal
# from rs.certify import certify

import os
import argparse

import models
#from utils import progress_bar

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--model', '-a', metavar='MODEL', default='resnet',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: alexnet)')
parser.add_argument('--depth', default=20, type=int,
                    help='depth for resnet')
parser.add_argument('--save_path', type=str, default='./results')
parser.add_argument('--task', default='train',
                    type=str, help='Task: train or test')
##########################################################################
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset')
parser.add_argument('--epochs', default=440, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=128, type=int,
                    help='batch size')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--lr_decay_ratio', default=0.1, type=float,
                    help='learning rate decay ratio')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: None)')
parser.add_argument('--optimizer', default='SGD', type=str, metavar='OPT',
                    help='optimizer function used')
parser.add_argument('--resume', default='True', type=str, help='resume from checkpoint')
###############################################################################
parser.add_argument('--training_method', default='macer', type=str, metavar='training method',
                    help='The method of training')
parser.add_argument('--lr_sigma', default=0.01, type=float, help='learning rate')
parser.add_argument('--gauss_num', default=16, type=int,
                    help='Number of Gaussian samples per input')
parser.add_argument('--sigma', default=0.25, type=float,
                    metavar='W', help='initial sigma for each data')
parser.add_argument('--lam', default=12.0, type=float,
                    metavar='W', help='initial sigma for each data')
parser.add_argument('--gamma', default=8.0, type=float, help='Hinge factor')
parser.add_argument('--beta', default=16.0, type=float, help='Inverse temperature of softmax (also used in test)')


def main():
    global args
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.save = args.optimizer + '_' + args.model + '_' + args.dataset + '_' + '_' + args.training_method + '_' \
                + str(args.sigma) + '_' + str(args.lam)
    save_path = os.path.join(args.save_path, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    logging.info("creating model %s", args.model)
    model = models.__dict__[args.model]
    model_config = {'input_size': 32, 'dataset': args.dataset, 'depth': args.depth}

    model = model(**model_config)

    if device == 'cuda':
        model = model.to(device)
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    print("created model with configuration: %s", model_config)
    print("run arguments: %s", args)
    with open(save_path+'/log.txt', 'a') as f:
        f.writelines(str(args) + '\n')

    # best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    num_parameters = sum([l.nelement() for l in model.parameters()])
    print("number of parameters: {}".format(num_parameters))
    # Data
    print('==> Preparing data..')
    if args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.24703225141799082, 0.24348516474564, 0.26158783926049628)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.24703225141799082, 0.24348516474564, 0.26158783926049628)),
        ])

    elif args.dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.5070588235294118, 0.48666666666666664, 0.4407843137254902),
            #                      (0.26745098039215687, 0.2564705882352941, 0.27607843137254906)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5070588235294118, 0.48666666666666664, 0.4407843137254902),
            #                      (0.26745098039215687, 0.2564705882352941, 0.27607843137254906)),
        ])

    elif args.dataset == 'svhn':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

    elif args.dataset == 'mnist':
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.5),
            #                      (0.5)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0:wq.5),
            #                      (0.5)),
        ])
    else:
        raise ValueError('No such dataset')

    # data_size = 0
    # for _, (inputs, targets) in enumerate(trainloader):
    #     data_size += targets.size(0)

    sigma = args.sigma * torch.ones(50000)
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = MultiStepLR(optimizer, milestones=[200, 400], gamma=args.lr_decay_ratio)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = MultiStepLR(optimizer, milestones=[200, 400], gamma=args.lr_decay_ratio)

    if args.resume == 'True':
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        if os.path.exists(save_path + '/ckpt.t7'):#, 'Error: no results directory found!'
            checkpoint = torch.load(save_path + '/ckpt.t7')
            model.load_state_dict(checkpoint['model'])
            start_epoch = checkpoint['epoch'] + 1
            if checkpoint['sigma'] is not None:
                sigma = checkpoint['sigma']
            scheduler.step(start_epoch)

    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        trainset = utils.create_set(trainset, sigma)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        # testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

    elif args.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        trainset = utils.create_set(trainset, sigma)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        # testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    elif args.dataset == 'svhn':
        trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)
        trainset = utils.create_set(trainset, sigma)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
        # testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    elif args.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
        trainset = utils.create_set(trainset, sigma)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
        # testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    else:
        raise ValueError('There is no such dataset')

    num_classes = 10
    train_vector = []

    if args.task == 'train':
        for epoch in range(start_epoch, args.epochs):
            strat_time = time.time()
            lr = optimizer.param_groups[0]['lr']
            print('create an optimizer with learning rate as:', lr)
            scheduler.step()
            model.train()
            c_loss, r_loss, acc = macer_train(args.lam, args.gauss_num, args.beta, args.gamma, args.lr_sigma,
                                              num_classes, model, trainloader, optimizer, device)

            print('Training time for each epoch is %g, optimizer is %s, model is %s' % (
                time.time() - strat_time, args.optimizer, args.model + str(args.depth)))

            if epoch % 20 == 0 and epoch >= 200:
                # Certify test
                print('===test(epoch={})==='.format(epoch))
                t1 = time.time()
                model.eval()
                certify(model, device, testset, transform_test, num_classes,
                        mode='hard', start_img=args.start_img, num_img=args.num_img,
                        sigma=args.sigma, beta=args.beta,
                        matfile=(None if save_path is None else os.path.join(save_path, '{}.mat'.format(epoch))))
                t2 = time.time()
                print('Elapsed time: {}'.format(t2 - t1))

            print('\n Epoch: {0}\t'
                         'Cross Entropy Loss {c_loss:.4f} \t'
                         'Robust Loss {r_loss:.3f} \t'
                         'Accuracy {acc:.4f} \n'
                         .format(epoch + 1, c_loss=c_loss, r_loss=r_loss,
                                 acc=acc))

            with open(save_path + '/log.txt', 'a') as f:
                f.write(str('\n Epoch: {0}\t'
                         'Cross Entropy Loss {c_loss:.4f} \t'
                         'Robust Loss {r_loss:.3f} \t'
                         'Accuracy {acc:.4f} \n'
                         .format(epoch + 1, c_loss=c_loss, r_loss=r_loss,
                                 acc=acc)) + '\n')

            state = {
                'model': model.state_dict(),
                'epoch': epoch,
                'sigma': torch.tensor([i[2] for i in trainset])
            }

            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            torch.save(state, save_path + '/ckpt.t7')
            if os.path.exists(save_path + '/train_vector'):
                with open(save_path + '/train_vector', 'rb') as fp:
                    train_vector = pickle.load(fp)
            train_vector.append([epoch, c_loss, r_loss, acc])
            with open(save_path + '/train_vector', 'wb') as fp:
                pickle.dump(train_vector, fp)

    else:
        certify(model, device, testset, num_classes,
                mode='both', start_img=args.start_img, num_img=args.num_img, skip=args.skip,
                sigma=args.sigma, beta=args.beta,
                matfile=(None if save_path is None else os.path.join(save_path, 'test.mat')))


# Training
def macer_train(lbd, gauss_num, beta, gamma, lr_sigma, num_classes, model, trainloader, optimizer, device):
    m = Normal(torch.tensor([0.0]).to(device),
               torch.tensor([1.0]).to(device))
    cl_total = 0.0
    rl_total = 0.0
    data_size = 0
    correct = 0

    if args.training_method == 'macer':
        for batch_idx, (inputs, targets, sigma) in enumerate(trainloader):
            inputs, targets, sigma_this_batch = inputs.to(device), targets.to(device), sigma.to(device)
            batch_size = len(inputs)
            data_size += targets.size(0)

            new_shape = [batch_size * gauss_num]
            new_shape.extend(inputs[0].shape)
            inputs = inputs.repeat((1, gauss_num, 1, 1)).view(new_shape)
            noise = torch.randn_like(inputs, device=device)
            for i in range(len(inputs.size()) - 1):
                sigma_this_batch.data = sigma_this_batch.data.unsqueeze(1)

            # sigma_this_batch_tmp = torch.zeros_like(sigma_this_batch)
            # sigma_this_batch_tmp.data.copy_(sigma_this_batch)
            sigma_this_batch.requires_grad_(True)

            for i in range(batch_size):
                noise[i * gauss_num: (i + 1) * gauss_num] *= sigma_this_batch[i]

            # for i in range(len(inputs.size()) - 1):
            #     sigma_this_batch.data = sigma_this_batch.data.squeeze(1)

            # inputs, noise = inputs.view(new_shape), noise.view(new_shape)
            noisy_inputs = inputs + noise

            outputs = model(noisy_inputs)
            outputs = outputs.reshape((batch_size, gauss_num, num_classes))

            # Classification loss
            outputs_softmax = F.softmax(outputs, dim=2).mean(1)
            outputs_logsoftmax = torch.log(outputs_softmax + 1e-10)  # avoid nan
            classification_loss = F.nll_loss(outputs_logsoftmax, targets, reduction='sum')
            cl_total += classification_loss.item()

            # Robustness loss
            beta_outputs = outputs * beta  # only apply beta to the robustness loss
            beta_outputs_softmax = F.softmax(beta_outputs, dim=2).mean(1)
            _, predicted = beta_outputs_softmax.max(1)
            correct += predicted.eq(targets).sum().item()

            top2 = torch.topk(beta_outputs_softmax, 2)
            top2_score = top2[0]
            top2_idx = top2[1]
            indices_correct = (top2_idx[:, 0] == targets)  # G_theta

            out0, out1 = top2_score[indices_correct, 0], top2_score[indices_correct, 1]
            robustness_loss = m.icdf(out1) - m.icdf(out0)
            indices = ~torch.isnan(robustness_loss) & ~torch.isinf(
                robustness_loss) & (torch.abs(robustness_loss) <= gamma)  # hinge
            out0, out1 = out0[indices], out1[indices]

            indices_correct = utils.cal_index(indices_correct, indices)
            robustness_loss = m.icdf(out1) - m.icdf(out0) + gamma
            robustness_loss = (robustness_loss * sigma_this_batch[indices_correct]).sum() / 2
            rl_total += robustness_loss.item()

            # Final objective function
            loss = classification_loss + lbd * robustness_loss
            loss /= batch_size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for i in range(len(inputs.size()) - 1):
                sigma_this_batch.grad.data = sigma_this_batch.grad.data.squeeze(1)
            sigma[indices_correct].data -= lr_sigma * sigma_this_batch.grad[indices_correct].cpu().data
            sigma_this_batch.grad.data.zero_()

        cl_total /= data_size
        rl_total /= data_size
        acc = 100 * correct / data_size

        return cl_total, rl_total, acc

    else:
        for batch_idx, (inputs, targets, sigma) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model.forward(inputs)
            loss = nn.CrossEntropyLoss(reduction='sum')(outputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            cl_total += loss.item()
            _, predicted= outputs.max(1)
            data_size += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        cl_total /= data_size
        acc = 100 * correct / data_size

        return cl_total, rl_total, acc


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    set_seed(888)
    main()
