# DiffusionModelAttack

Hardware Requirements: 1x high-end NVIDIA GPU with at least 16GB memory

Software Requirements: Python: 3.8, CUDA: 11.3, cuDNN: 8.4.1

To install other requirements:

`pip install -r requirements.txt`

The dataset used in this project is 'handpose_x_gesture_v1' from https://codechina.csdn.net/EricLee/classification.

Use Train_main.py to train an initial ResNet.

Use main.py to generate adversarial samples based on diffusion model.

Modify Train_main.py to implent adversarial defence.

Use cal_acc to calculate the classification accuracy.
