# Data_Science_Bowl_2018

# Dataset
We used Data Science Bowl 2018 dataset which contains large unseen microscopy images. The dataset is divided into three kinds of dataset. They are 30,131 training data (image + mask), 65 stage_1 testing data, and 3,020 stage_2 testing data. In this competition, we only used the training data and stage_2 testing data.

# Proposed 1: Unet
In this part, we used a normal Unet architecture with 23 convolutional layers and followed by ReLU activation function. The architecture was divided into contracting path and expansive path. So, the contracting path consists of two 3x3 convolutional layers, a ReLU, a 2x2 max pooling layer, and a dropout layer repeatedly. In contrast, the expansive path consists of a 2x2 transpose convolutional layer, a concatenation of a feature map from contracting path, two 3x3 convolutional layers, and a ReLU, repeatedly. For the output, sigmoid was used.

Before we feed the data to the network, training dataset was augmented including the image and mask data. Data augmentation is essential when the training dataset is too few to teach the network the invariance properties. We added shear (0.5), rotation (50), zoom (0.2), width shift (0.2), height shift (0.2), and reflect. Besides that, we divided the main dataset into 90% training dataset and 10% validation dataset.

Moreover, the batch size was set to 10. For the loss function, we used binary cross-entropy. In the original paper, they used stochastic gradient descent for the optimizer but we used Adam because it learned faster. We trained the model with 30 epochs and early stopping was also be used with patience of 3 epochs. So, the training would be stopped immediately when there was no improvement of the validation loss in 3 epochs. Then, we tested the weight to data science bowl 2018 dataset, we got mAP score of 0.20102.

# Proposed 2: Unet + VGG16
In this experiment, we used a normal Unet like we did in proposed 1 but we didn’t use the dropout layer. We also concatenated VGG16 application as model with weights pre-trained on ImageNet in the expansive path. In addition, the parameter and the data augmentation used were similar to what we did in proposed 1. With VGG16 model, the result was slightly improved to 0.21382.

# Proposed 3: Mask RCNN + Resnet50 + Weak Augmentation
In this proposal, we tried to use Mask RCNN, an instance segmentation approach. For the parameter, we set some parameters that should be suitable for the training task with the data science bowl 2018 dataset.

We set the learning rate to 0.00003 to make sure that the training process worked well per epoch and avoided overshoot which happens when the learning rate is too high. Besides that, we added weight decay to prevent overfit and the value was based on Faster-RCNN original paper. We also set some threshold parameter to make sure the detection task was precise.

For the data augmentation, we only used some of vertical and horizontal flip, weak rotate augmentation, multiply, and gaussian blur. The main contribution of this proposal is we used Mask RCNN, Resnet50 backbone, and weak augmentation. We trained the model with 40 epochs (20 epochs for heads layer and the other 20 epochs for all layers). ImageNet model was used as initial weight. When we tested the result, we got mAP of 0.45707 on 40th epoch.

|   Parameters  |   Proposed 3  |   Proposed 4  |
| ------------- | ------------- | ------------- |
| Learning rate  | 0.00003  | 0.00003  |
| Momentum  | 0.9  | 0.9  |
| Weight Decay  | 0.0001  | 0.0001  |
| Mean Pixel  | [123.7, 116.8, 103.9]  | [123.7, 116.8, 103.9]  |
| Detection Min Confidence  | 0.9  | 0.9  |
| Detection NMS Threshold  | 0.2  | 0.2  |
| RPN NMS Threshold  | 0.9  | 0.9  |
| RPN Anchor Scale  | [8, 16, 32, 64, 128]  | [8, 16, 32, 64, 128] |
| RPN Train Anchors Per Image  | 256 | 256  |
| Image Min DIM  | 512  | 512 |
| Image Max DIM  | 512  | 512 |
| Backbone  | Resnet50  | Resnet101  |
| Model  | ImageNet  | Coco |
| Use Mini Mask  | True (56, 56)  | True (56, 56)  |
| Data Augmentation  | Some of (Flip Vertical and Horizontal (0.5), One of (Multi Rotate (90, 180, 270), Multiply, Gaussian Blur))  | Multi Flip Vertical and Horizontal, Multi Rotate (45, 90, 180, 270), Scale 50%-150%, Sometimes (Scale 80%-120%, Translate, Shear)  |

# Proposed 4: Mask RCNN + Resnet101 + Strong Augmentation
In proposed 4, we used the same parameter as proposed 3 but we changed the backbone to Resnet101 and trained using the coco initial weight. We could see the parameter in Table X. Other than that, we set the training epoch to 80 (20 epochs for heads layer and 60 epochs for all layers). While training, the weights per epoch were saved. We got the best result on 75th epoch with mAP score of 0.53678.

The reason why we chose the weight on 75th epoch was because it has the lowest training loss as well as validation loss. The result was proved in Table 2. We found that the validation dataset has high similarity with test dataset because decreasing validation loss made better mAP test and vice versa.

# Results

|   Methods  |   mAP  |
| ------------- | ------------- |
| Unet (Proposed 1)  | 0.20102  |
| Unet + VGG16 (Proposed 2)  | 0.21382 |
| Baseline  | 0.43186 |
| Mask RCNN + Resnet50 + Weak Augmentation (Proposed 3)  | 0.45707  |
| Mask RCNN + Resnet50 + Strong Augmentation (Addition)  | 0.48939  |
| Mask RCNN + Resnet101 + Weak Augmentation (Addition)  | 0.46878  |
| Mask RCNN + Resnet101 + Strong Augmentation (Proposed 4)  | **0.53678**  |

![Gif](https://user-images.githubusercontent.com/31305502/71643319-4ab0d780-2cf3-11ea-92cd-526074ee1c26.gif)
