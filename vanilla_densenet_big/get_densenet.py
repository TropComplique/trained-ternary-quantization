import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim
from torch.nn.init import normal
from densenet import DenseNet
import torch.utils.model_zoo as model_zoo
from PIL import Image


NUM_CLASSES = 200


def resize_filter(f):
    """Treats a filter like a
    small image and resizes it.

    Arguments:
        f: a numpy float array with shape
            [filter_size, filter_size, 3].
    Returns:
        a numpy float array with shape
            [new_size, new_size, 3],
            where new_size = 3.
    """
    min_val = f.min()
    max_val = f.max()

    # convert to [0, 255] range
    f = (f - min_val)/(max_val - min_val)
    f *= 255.0

    img = Image.fromarray(f.astype('uint8'))
    img = img.resize((3, 3), Image.LANCZOS)
    f = np.asarray(img, dtype='float32')/255.0

    # back to the original range
    f *= (max_val - min_val)
    f += min_val

    return f


def get_model():
    """Get the model, the loss, and an optimizer"""

    # get DenseNet-121
    model = DenseNet()

    state_dict = model_zoo.load_url(
        'https://download.pytorch.org/models/densenet121-241335ed.pth'
    )

    # in the original pretrained model the first conv layer has
    # 64 filters with size (7, 7), but because i use
    # images of smaller size i resize filters to (3, 3)
    first_conv = state_dict['features.conv0.weight'].cpu().numpy()
    n_filters = first_conv.shape[0]
    new_filters = np.zeros((n_filters, 3, 3, 3), 'float32')
    for i, f in enumerate(first_conv):
        f = f.transpose(1, 2, 0)
        f = resize_filter(f)
        f = f.transpose(2, 0, 1)
        new_filters[i] = f
    state_dict['features.conv0.weight'] = torch.FloatTensor(new_filters).cuda()

    # reset weights of the last fc layer
    weight = torch.zeros(NUM_CLASSES, 1024)
    normal(weight, std=0.01)
    state_dict['classifier.weight'] = weight.cuda()
    state_dict['classifier.bias'] = torch.zeros(NUM_CLASSES).cuda()

    # load pretrained model
    model.load_state_dict(state_dict)

    # make all params not trainable
    for p in model.parameters():
        p.requires_grad = False

    # make the last fc layer trainable
    model.classifier.weight.requires_grad = True
    model.classifier.bias.requires_grad = True

    # make the last batch norm layer trainable
    model.features.norm5.weight.requires_grad = True
    model.features.norm5.bias.requires_grad = True

    # create different parameter groups
    weights = [model.classifier.weight]
    biases = [model.classifier.bias]
    bn_weights = [model.features.norm5.weight]
    bn_biases = [model.features.norm5.bias]

    params = [
        {'params': weights, 'weight_decay': 1e-4},
        {'params': biases},
        {'params': bn_weights},
        {'params': bn_biases}
    ]
    optimizer = optim.Adam(params, lr=1e-3)

    loss = nn.CrossEntropyLoss().cuda()
    model = model.cuda()  # move the model to gpu
    return model, loss, optimizer
