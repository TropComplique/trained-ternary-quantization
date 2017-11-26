# Trained Ternary Quantization
`pytorch` implementation of [Trained Ternary Quantization](https://arxiv.org/abs/1612.01064), a way of replacing full precision weights of a neural network by ternary values. I tested it on [Tiny ImageNet](https://tiny-imagenet.herokuapp.com/) dataset. The dataset consists of 64x64 images and has 200 classes.

The quantization roughly proceeds as follows.
1. Train a model of your choice as usual (or take a trained model).
2. Copy all full precision weights that you want to quantize. Then do the initial quantization:  
in the model replace them by ternary values {-1, 0, +1} using some heuristic.
3. Repeat until convergence:
   * Make the forward pass with the quantized model.
   * Compute gradients for the quantized model.
   * Preprocess the gradients and apply them to the copy of full precision weights.
   * Requantize the model using the changed full precision weights.
4. Throw away the copy of full precision weights and use the quantized model.

## Results
I believe that this results can be made better by spending more time on hyperparameter optimization.

| model | accuracy, % | top5 accuracy, % | number of parameters |
| --- | --- | --- | --- |
| DenseNet-121 | 74 | 91 | 7151176 |
| TTQ DenseNet-121 | 66 | 87 | ~7M 2-bit, 88% are zeros |
| small DenseNet | 49 | 75 | 440264 |
| TTQ small DenseNet | 40 | 67 | ~0.4M 2-bit, 38% are zeros |
| SqueezeNet | 52 | 77 | 827784 |
| TTQ SqueezeNet | 36 | 63 | ~0.8M 2-bit, 66% are zeros |

## Implementation details
* I use pretrained DenseNet-121, but I train SqueezeNet and small DenseNet from scratch.
* I modify the SqueezeNet architecture by adding batch normalizations and skip connections.
* I quantize all layers except the first CONV layer, the last FC layer, and all BATCH_NORM layers.


## How to reproduce results
For example, for small DenseNet:
1. Download [Tiny ImageNet](https://tiny-imagenet.herokuapp.com/) dataset and extract it to `~/data` folder.
2. Run `python utils/move_tiny_imagenet_data.py` to prepare the data.
3. Go to `vanilla_densenet_small/`. Run `train.ipynb` to train the model as usual.  
Or you can skip this step and use `model.pytorch_state` (the model already trained by me).
4. Go to `ttq_densenet_small/`.
5. Run `train.ipynb` to do TTQ.
6. Run `test_and_explore.ipynb` to explore the quantized model.

To use this on your data you need to edit `utils/input_pipeline.py` and to change
the model architecture in files like `densenet.py` and `get_densenet.py` as you like.

## Requirements
* pytorch 0.2, Pilllow, torchvision
* numpy, sklearn, tqdm, matplotlib
