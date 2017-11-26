# Summary
Here I finetuned a pretrained DenseNet-121. Tiny ImageNet consists of 64x64 images,
but the pretrained model expects much bigger images (224x224). So before finetuning
I made some changes to the network architecture.

## Preparing the model for finetuning
1. Changed the first CONV layer: resized filters (7 -> 3), reduced the stride (2 -> 1).  
To resize filters I treated them like small 7x7 images and then used `PIL.Image.resize`.
2. Removed the first pooling layer.
3. Randomly reinitialized the last FC layer.

## Training steps

| step | layers | optimizer | epochs | time, in hours | accuracy, % |
| --- | --- | --- | --- | --- | --- |
| 1 | only the last FC layer | Adam, lr=1e-3 | 5 | 1 | 45 |
| 2 | the whole network | SGD with nesterov=0.9, lr=1e-5 | 5 | 3 | 65 |
| 3 | the whole network | lr=1e-4 | 5 | 3 | 71 |
| 4 | the whole network | lr=1e-4 | 5 | 3 | 72.8 |
| 5 | the whole network | lr=5e-5 | 3 | 1.5 | 73.6 |
