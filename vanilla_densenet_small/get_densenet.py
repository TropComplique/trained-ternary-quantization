import torch.nn as nn
import torch.optim as optim
from torch.nn.init import constant, kaiming_uniform
from densenet import DenseNet


def get_model():

    model = DenseNet(
        growth_rate=12, block_config=(8, 12, 10),
        num_init_features=48, bn_size=4, drop_rate=0.25,
        final_drop_rate=0.25, num_classes=200
    )

    # create different parameter groups
    weights = [
        p for n, p in model.named_parameters()
        if 'conv' in n or 'classifier.weight' in n
    ]
    biases = [model.classifier.bias]
    bn_weights = [
        p for n, p in model.named_parameters()
        if 'norm' in n and 'weight' in n
    ]
    bn_biases = [
        p for n, p in model.named_parameters()
        if 'norm' in n and 'bias' in n
    ]

    # parameter initialization
    for p in weights:
        kaiming_uniform(p)
    for p in biases:
        constant(p, 0.0)
    for p in bn_weights:
        constant(p, 1.0)
    for p in bn_biases:
        constant(p, 0.0)

    params = [
        {'params': weights, 'weight_decay': 1e-4},
        {'params': biases},
        {'params': bn_weights},
        {'params': bn_biases}
    ]
    optimizer = optim.SGD(params, lr=1e-1, momentum=0.95, nesterov=True)

    loss = nn.CrossEntropyLoss().cuda()
    model = model.cuda()  # move the model to gpu
    return model, loss, optimizer
