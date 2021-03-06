import torch
import utils
from torch.distributions.normal import Normal
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import time

def macer_train(method, sigma_net, logsub, lbd, gauss_num, beta, gamma, lr_sigma, num_classes, model, trainset,
                batch_sampler, optimizer, device, epoch, average='False'):
    m = Normal(torch.tensor([0.0]).to(device),
               torch.tensor([1.0]).to(device))

    cl_total = 0.0
    rl_total = 0.0
    data_size = 0
    correct = 0
    _, sigma_total = trainset
    sigma_mean = sigma_total.mean()
    index_tmp = torch.tensor([0] * len(sigma_total), dtype=torch.bool).to(device)
    if method == 'macer':
        if epoch >= 0:
            lr_sigma = lr_sigma
        else:
            lr_sigma = 0.0

        if sigma_net is not None:
            optimizer_sigma = optim.SGD(sigma_net.parameters(), lr=lr_sigma, weight_decay=5e-4)
        else:
            optimizer_sigma = None

        for batch_idx, (inputs, targets, index) in enumerate(batch_sampler):
            time_1 = time.time()
            inputs, targets, index = inputs.to(device), targets.to(device), index.to(device)
            if sigma_net is None:
                sigma_this_batch = sigma_total.index_select(0, index)
               # sigma_this_batch.requires_grad_(True)
               
            else:
                sigma_this_batch = sigma_net.forward(inputs, average)

            batch_size = len(inputs)
            data_size += targets.size(0)

            new_shape = [batch_size * gauss_num]
            new_shape.extend(inputs[0].shape)
            inputs = inputs.repeat((1, gauss_num, 1, 1)).view(new_shape)
            noise = torch.randn_like(inputs, device=device)
            inputs.requires_grad_(True)
            for i in range(batch_size):
                noise[i * gauss_num: (i + 1) * gauss_num] *= sigma_this_batch[i]

            noisy_inputs = inputs + noise
           # noisy_inputs.requires_grad_(True)
            outputs = model(noisy_inputs)
            outputs = outputs.reshape((batch_size, gauss_num, num_classes))

            # Classification loss
            outputs_softmax = F.softmax(outputs, dim=2).mean(1)
            outputs_logsoftmax = torch.log(outputs_softmax + 1e-10)  # avoid nan
            classification_loss = F.nll_loss(outputs_logsoftmax, targets, reduction='sum')
            cl_total += classification_loss.item()
            _, predicted = outputs_softmax.max(1)
            correct += predicted.eq(targets).sum().item()
            # print(classification_loss)

            if epoch <= 90:
                loss = classification_loss

            else:
                beta_outputs = outputs * beta  # only apply beta to the robustness loss
                beta_outputs_softmax = F.softmax(beta_outputs, dim=2).mean(1)

                top2 = torch.topk(beta_outputs_softmax, 2)
                top2_score = top2[0]
                top2_idx = top2[1]
                indices_correct = (top2_idx[:, 0] == targets)  # G_theta

                out0_correct, out1_correct = top2_score[indices_correct, 0], top2_score[indices_correct, 1]

                out0_correct, out1_correct = torch.clamp(out0_correct, 0, 0.9999999), torch.clamp(out1_correct, 1e-7, 1)

                if logsub == 'True':
                    robustness_loss_correct = torch.clamp(m.icdf(out0_correct) - m.icdf(out1_correct), 0, gamma)  # + gamma

                    robustness_loss_correct = torch.log(
                        1 + torch.exp(robustness_loss_correct * sigma_this_batch[indices_correct] / 2)).sum()

                    robustness_loss = robustness_loss_correct

                else:
                    robustness_loss_correct = m.icdf(out0_correct) - m.icdf(out1_correct)
                    robustness_loss_correct = torch.clamp(robustness_loss_correct, 0, gamma)  # + gamma

                    robustness_loss = (robustness_loss_correct * sigma_this_batch[indices_correct]).sum() / 2
                    rl_total += lbd * robustness_loss.item()

            # Final objective function
                loss = classification_loss - lbd * robustness_loss

            loss /= batch_size
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if sigma_net is not None:
                optimizer_sigma.step()
                optimizer_sigma.zero_grad()
            else:
                index = list(index.cpu().numpy())
                for i in range(batch_size):
                    sigma_grad = inputs.grad[i * gauss_num: (i + 1) * gauss_num] * noise[i * gauss_num: (i + 1) * gauss_num]
                    sigma_grad = 4 * sigma_grad.sum()
                    sigma_this_batch.data[i] -= lr_sigma * sigma_grad.data
               # print(sigma_this_batch)    
                sigma = torch.max(1e-8 * torch.ones_like(sigma_this_batch), sigma_this_batch.detach())
                index_select = utils.gen_index(index_tmp, index)
                sigma_total[index_select] = sigma
                index_tmp = utils.recover_index(index_tmp, index)
               # print(time.time() - time_1)
                inputs.detach()
        if average != 'False':
            trainset[1] = sigma_total - sigma_total.mean() + sigma_mean
        cl_total /= data_size
        rl_total /= data_size
        acc = 100 * correct / data_size

        return cl_total, rl_total, acc

    else:
        for batch_idx, (inputs, targets, index) in enumerate(batch_sampler):
            time_1 = time.time()
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model.forward(inputs)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            cl_total += loss.item() * len(inputs)
            _, predicted = outputs.max(1)
            data_size += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            print(time.time() - time_1)
        cl_total /= data_size
        acc = 100 * correct / data_size

        return cl_total, rl_total, acc
