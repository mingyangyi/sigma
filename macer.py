import torch
import utils
from torch.distributions.normal import Normal
import torch.optim as optim
import torch.nn.functional as F


def macer_train(method, sigma_net, logsub, lbd, gauss_num, beta, gamma, lr_sigma, num_classes, model, trainset,
                batch_sampler, optimizer, device):
    m = Normal(torch.tensor([0.0]).to(device),
               torch.tensor([1.0]).to(device))
    cl_total = 0.0
    rl_total = 0.0
    data_size = 0
    correct = 0
    (inputs_total, targets_total, sigma_total) = trainset

    if method == 'macer':
        if sigma_net is not None:
            optimizer_sigma = optim.SGD(sigma_net.parameters(), lr=lr_sigma, weight_decay=5e-4)
        else:
            optimizer_sigma = None

        for batch_idx, index in enumerate(batch_sampler):
            inputs, targets = inputs_total.index_select(0, torch.tensor(index)).to(device), \
                              targets_total.index_select(0, torch.tensor(index)).to(device)

            if sigma_net is None:
                sigma_this_batch = sigma_total.index_select(0, torch.tensor(index)).to(device)
                sigma_this_batch.requires_grad_(True)

            else:
                sigma_this_batch = sigma_net.forward(inputs)

            batch_size = len(inputs)
            data_size += targets.size(0)

            new_shape = [batch_size * gauss_num]
            new_shape.extend(inputs[0].shape)
            inputs = inputs.repeat((1, gauss_num, 1, 1)).view(new_shape)
            noise = torch.randn_like(inputs, device=device)

            for i in range(batch_size):
                noise[i * gauss_num: (i + 1) * gauss_num] *= sigma_this_batch[i]

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

            if logsub is 'True':
                robustness_loss = out0 * torch.log(out1 + 1.0)
                robustness_loss = (robustness_loss * sigma_this_batch[indices_correct]).sum() / \
                                  robustness_loss.size()[0].item() * batch_size
            else:
                robustness_loss = m.icdf(out1) - m.icdf(out0)
                indices = ~torch.isnan(robustness_loss) & ~torch.isinf(
                    robustness_loss) & (torch.abs(robustness_loss) <= gamma)  # hinge
                indices_correct = utils.cal_index(indices_correct, indices)
                out0, out1 = out0[indices], out1[indices]
                robustness_loss = m.icdf(out1) - m.icdf(out0) + gamma
                robustness_loss = (robustness_loss * sigma_this_batch[indices_correct]).sum() / 2

            rl_total += robustness_loss.item()

            # Final objective function
            loss = classification_loss + lbd * robustness_loss
            loss /= batch_size
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if sigma_net is not None:
                optimizer_sigma.step()
                optimizer_sigma.zero_grad()
            else:
                sigma_this_batch.requires_grad_(False)
                sigma_this_batch[indices_correct] -= lr_sigma * sigma_this_batch.grad[indices_correct]
                sigma_this_batch.grad.data.zero_()
                sigma = torch.max(1e-8 * torch.ones_like(sigma_this_batch), sigma_this_batch.detach())
                index = utils.gen_index(index, len(sigma_total))
                sigma_total[index] = sigma.cpu()

        cl_total /= data_size
        rl_total /= data_size
        acc = 100 * correct / data_size

        return cl_total, rl_total, acc

    else:
        for batch_idx, index in enumerate(batch_sampler):
            inputs, targets, sigma = inputs.index_select(0, torch.tensor(index)).to(device), \
                                     targets.index_select(0, torch.tensor(index)).to(device), \
                                     sigma_total.index_select(0, torch.tensor(index)).to(device)
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