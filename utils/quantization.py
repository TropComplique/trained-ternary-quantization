import torch
from torch.autograd import Variable
import torch.nn.functional as F
from .training import _accuracy


def initial_scales(kernel):
    return 1.0, 1.0


def quantize(kernel, w_p, w_n, t):
    """
    Return quantized weights of a layer.
    Only possible values of quantized weights are: {zero, w_p, -w_n}.
    """
    delta = t*kernel.abs().max()
    a = (kernel > delta).float()
    b = (kernel < -delta).float()
    return w_p*a + (-w_n*b)


def get_grads(kernel_grad, kernel, w_p, w_n, t):
    """
    Arguments:
        kernel_grad: gradient with respect to quantized kernel.
        kernel: corresponding full precision kernel.
        w_p, w_n: scaling factors.
        t: hyperparameter for quantization.

    Returns:
        1. gradient for the full precision kernel.
        2. gradient for w_p.
        3. gradient for w_n.
    """
    delta = t*kernel.abs().max()
    # masks
    a = (kernel > delta).float()
    b = (kernel < -delta).float()
    c = torch.ones(kernel.size()).cuda() - a - b
    # scaled kernel grad and grads for scaling factors (w_p, w_n)
    return w_p*a*kernel_grad + w_n*b*kernel_grad + 1.0*c*kernel_grad,\
        (a*kernel_grad).sum(), (b*kernel_grad).sum()


def optimization_step(model, loss, x_batch, y_batch, optimizer_list, t):
    """Make forward pass and update model parameters with gradients."""

    # parameter 't' is a hyperparameter for quantization

    # 'optimizer_list' contains optimizers for
    # 1. full model (all weights including quantized weights),
    # 2. backup of full precision weights,
    # 3. scaling factors for each layer
    optimizer, optimizer_fp, optimizer_sf = optimizer_list

    x_batch, y_batch = Variable(x_batch.cuda()), Variable(y_batch.cuda(async=True))
    # forward pass using quantized model
    logits = model(x_batch)

    # compute logloss
    loss_value = loss(logits, y_batch)
    batch_loss = loss_value.data[0]

    # compute accuracies
    pred = F.softmax(logits)
    batch_accuracy, batch_top5_accuracy = _accuracy(y_batch, pred, top_k=(1, 5))

    optimizer.zero_grad()
    optimizer_fp.zero_grad()
    optimizer_sf.zero_grad()
    # compute grads for quantized model
    loss_value.backward()

    # get all quantized kernels
    all_kernels = optimizer.param_groups[1]['params']

    # get their full precision backups
    all_fp_kernels = optimizer_fp.param_groups[0]['params']

    # get two scaling factors for each quantized kernel
    scaling_factors = optimizer_sf.param_groups[0]['params']

    for i in range(len(all_kernels)):

        # get a quantized kernel
        k = all_kernels[i]

        # get corresponding full precision kernel
        k_fp = all_fp_kernels[i]

        # get scaling factors for the quantized kernel
        f = scaling_factors[i]
        w_p, w_n = f.data[0], f.data[1]

        # get modified grads
        k_fp_grad, w_p_grad, w_n_grad = get_grads(k.grad.data, k_fp.data, w_p, w_n, t)

        # grad for full precision kernel
        k_fp.grad = Variable(k_fp_grad)

        # we don't need to update the quantized kernel directly
        k.grad.data.zero_()

        # grad for scaling factors
        f.grad = Variable(torch.FloatTensor([w_p_grad, w_n_grad]).cuda())

    # update all non quantized weights in quantized model
    # (usually, these are the last layer, the first layer, and all batch norm params)
    optimizer.step()

    # update all full precision kernels
    optimizer_fp.step()

    # update all scaling factors
    optimizer_sf.step()

    # update all quantized kernels with updated full precision kernels
    for i in range(len(all_kernels)):

        k = all_kernels[i]
        k_fp = all_fp_kernels[i]
        f = scaling_factors[i]
        w_p, w_n = f.data[0], f.data[1]

        # requantize a quantized kernel using updated full precision weights
        k.data = quantize(k_fp.data, w_p, w_n, t)

    return batch_loss, batch_accuracy, batch_top5_accuracy
