import torch.nn as nn
import torch.optim as optim

import sys
sys.path.append('../vanilla_squeezenet/')
from squeezenet import SqueezeNet


def get_model(learning_rate=1e-3):

    model = SqueezeNet()
    
    # set the first layer not trainable
    model.features[0].weight.requires_grad = False

    # all conv layers except the first and the last
    all_conv_weights = [
        (n, p) for n, p in model.named_parameters()
        if 'weight' in n and not 'bn' in n and not 'features.1.' in n
    ]
    weights_to_be_quantized = [
        p for n, p in all_conv_weights
        if not ('classifier' in n or 'features.0.' in n)
    ]
    
    # the last layer
    weights = [model.classifier[1].weight]
    biases = [model.classifier[1].bias]
    
    # parameters of batch_norm layers
    bn_weights = [
        p for n, p in model.named_parameters()
        if ('bn' in n or 'features.1.' in n) and 'weight' in n
    ]
    bn_biases = [
        p for n, p in model.named_parameters()
        if ('bn' in n or 'features.1.' in n) and 'bias' in n
    ]

    params = [
        {'params': weights, 'weight_decay': 1e-4},
        {'params': weights_to_be_quantized},
        {'params': biases},
        {'params': bn_weights},
        {'params': bn_biases}
    ]
    optimizer = optim.Adam(params, lr=learning_rate)

    # loss function
    criterion = nn.CrossEntropyLoss().cuda()
    # move the model to gpu
    model = model.cuda()
    return model, criterion, optimizer
