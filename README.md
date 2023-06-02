# EuroSat
## Project Overview
In this project, a deep Convolutional Neural Network (CNNs) is built with PyTorch to classify Land use and cover dataset from Sentinel-2 satellite images.


![](https://raw.githubusercontent.com/phelber/EuroSAT/master/eurosat_overview_small.jpg)

## Dataset
[EuroSAT: Land Use and Land Cover Classification with Sentinel-2](https://github.com/phelber/eurosat)

The dataset comprises 27,000 labeled and geo-referenced images, divided into 10 distinct classes. It is available in two versions: RGB and multi-spectral. The RGB version consists of images encoded in JPEG format, representing the optical Red, Green, and Blue (RGB) frequency bands. These images provide color information in the visible spectrum. The multi-spectral version of the dataset includes all 13 Sentinel-2 bands, which retains the original value range of the Sentinel-2 bands, enabling access to a more comprehensive set of spectral information.

1. [RGB](https://madm.dfki.de/files/sentinel/EuroSAT.zip) (**The employed one in this project**)
2. [Multi-spectral](https://madm.dfki.de/files/sentinel/EuroSATallBands.zip)

## Workflow
##### 1. Import the relevant modules and libraries 
##### 2. Define a custom dataset class for the satellite Images.
##### 3. Builds a pre-trained VGG19 model with a custom classifier for image classification, select the Loss Function and the optimizer for training the model.
##### 4. Build a function to performs a single training iteration on every batch of data, and other function to calculates the accuracy of the model's predictions on a batch of data.
##### 5. Run a for training loop that runs for a specified number of epochs to iterate over the training and test dataset loaders, calculates the training loss, accuracy, and prints their values for each epoch.
##### 6. Visualize the training and test loss values as well as the training and test accuracy values over increasing epochs.

## Training
### The First Training
The first training is performed using:
1. Model : The Pretrained VGG19 Model 
2. Loss Function:  Cross-Entropy 
3. Optimizer: Adam
4. Learning Rate: 0.001
5. Apply Dropout: True
6. Number of training epochs: 25
7. Traing Data Size: 22000
8. Test Data Size: 5000
9. Model Summary: 

            Layer (type:depth-idx)                   Output Shape              Param 
            ==========================================================================================

            ├─Sequential: 1-1                        [-1, 512, 2, 2]           --
            |    └─Conv2d: 2-1                       [-1, 64, 64, 64]          (1,792)
            |    └─ReLU: 2-2                         [-1, 64, 64, 64]          --
            |    └─Conv2d: 2-3                       [-1, 64, 64, 64]          (36,928)
            |    └─ReLU: 2-4                         [-1, 64, 64, 64]          --
            |    └─MaxPool2d: 2-5                    [-1, 64, 32, 32]          --
            |    └─Conv2d: 2-6                       [-1, 128, 32, 32]         (73,856)
            |    └─ReLU: 2-7                         [-1, 128, 32, 32]         --
            |    └─Conv2d: 2-8                       [-1, 128, 32, 32]         (147,584)
            |    └─ReLU: 2-9                         [-1, 128, 32, 32]         --
            |    └─MaxPool2d: 2-10                   [-1, 128, 16, 16]         --
            |    └─Conv2d: 2-11                      [-1, 256, 16, 16]         (295,168)
            |    └─ReLU: 2-12                        [-1, 256, 16, 16]         --
            |    └─Conv2d: 2-13                      [-1, 256, 16, 16]         (590,080)
            |    └─ReLU: 2-14                        [-1, 256, 16, 16]         --
            |    └─Conv2d: 2-15                      [-1, 256, 16, 16]         (590,080)
            |    └─ReLU: 2-16                        [-1, 256, 16, 16]         --
            |    └─Conv2d: 2-17                      [-1, 256, 16, 16]         (590,080)
            |    └─ReLU: 2-18                        [-1, 256, 16, 16]         --
            |    └─MaxPool2d: 2-19                   [-1, 256, 8, 8]           --
            |    └─Conv2d: 2-20                      [-1, 512, 8, 8]           (1,180,160)
            |    └─ReLU: 2-21                        [-1, 512, 8, 8]           --
            |    └─Conv2d: 2-22                      [-1, 512, 8, 8]           (2,359,808)
            |    └─ReLU: 2-23                        [-1, 512, 8, 8]           --
            |    └─Conv2d: 2-24                      [-1, 512, 8, 8]           (2,359,808)
            |    └─ReLU: 2-25                        [-1, 512, 8, 8]           --
            |    └─Conv2d: 2-26                      [-1, 512, 8, 8]           (2,359,808)
            |    └─ReLU: 2-27                        [-1, 512, 8, 8]           --
            |    └─MaxPool2d: 2-28                   [-1, 512, 4, 4]           --
            |    └─Conv2d: 2-29                      [-1, 512, 4, 4]           (2,359,808)
            |    └─ReLU: 2-30                        [-1, 512, 4, 4]           --
            |    └─Conv2d: 2-31                      [-1, 512, 4, 4]           (2,359,808)
            |    └─ReLU: 2-32                        [-1, 512, 4, 4]           --
            |    └─Conv2d: 2-33                      [-1, 512, 4, 4]           (2,359,808)
            |    └─ReLU: 2-34                        [-1, 512, 4, 4]           --
            |    └─Conv2d: 2-35                      [-1, 512, 4, 4]           (2,359,808)
            |    └─ReLU: 2-36                        [-1, 512, 4, 4]           --
            |    └─MaxPool2d: 2-37                   [-1, 512, 2, 2]           --
            ├─AdaptiveAvgPool2d: 1-2                 [-1, 512, 1, 1]           --
            ├─Sequential: 1-3                        [-1, 10]                  --
            |    └─Flatten: 2-38                     [-1, 512]                 --
            |    └─Linear: 2-39                      [-1, 128]                 65,664
            |    └─ReLU: 2-40                        [-1, 128]                 --
            |    └─Dropout: 2-41                     [-1, 128]                 --
            |    └─Linear: 2-42                      [-1, 10]                  1,290
            ==========================================================================================
            Total params: 20,091,338
            Trainable params: 66,954
            Non-trainable params: 20,024,384
            Total mult-adds (G): 1.61
            ==========================================================================================
            Input size (MB): 0.05
            Forward/backward pass size (MB): 9.25
            Params size (MB): 76.64
            Estimated Total Size (MB): 85.94
            ==========================================================================================        
                            
                            
10. Results 
     ![image](https://github.com/MuhammedM294/EuroSat/assets/89984604/9ad8985b-e28b-4534-811a-e17c0b098195)
     

### The Second Training
The second training is performed using:
1. Model:

            def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
                return nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=1)
                )
            def build_model():
                model = nn.Sequential(
                    conv_block(3, 64,3),
                    conv_block(64, 128,3),
                    conv_block(128, 256,3),
                    conv_block(256, 512,3),
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(),
                    nn.Linear(512, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, 10),
                )


2. Loss Function:  Cross-Entropy 
3. Optimizer: Adam
4. Learning Rate: 0.001
5. Apply Dropout: True
6. Number of training epochs: 25
7. Traing Data Size: 22000
8. Test Data Size: 5000
9. Model Summary: 


            ==========================================================================================
            Layer (type:depth-idx)                   Output Shape              Param #
            ==========================================================================================
            ├─Sequential: 1-1                        [-1, 64, 63, 63]          --
            |    └─Conv2d: 2-1                       [-1, 64, 64, 64]          1,792
            |    └─BatchNorm2d: 2-2                  [-1, 64, 64, 64]          128
            |    └─ReLU: 2-3                         [-1, 64, 64, 64]          --
            |    └─MaxPool2d: 2-4                    [-1, 64, 63, 63]          --
            ├─Sequential: 1-2                        [-1, 128, 62, 62]         --
            |    └─Conv2d: 2-5                       [-1, 128, 63, 63]         73,856
            |    └─BatchNorm2d: 2-6                  [-1, 128, 63, 63]         256
            |    └─ReLU: 2-7                         [-1, 128, 63, 63]         --
            |    └─MaxPool2d: 2-8                    [-1, 128, 62, 62]         --
            ├─Sequential: 1-3                        [-1, 256, 61, 61]         --
            |    └─Conv2d: 2-9                       [-1, 256, 62, 62]         295,168
            |    └─BatchNorm2d: 2-10                 [-1, 256, 62, 62]         512
            |    └─ReLU: 2-11                        [-1, 256, 62, 62]         --
            |    └─MaxPool2d: 2-12                   [-1, 256, 61, 61]         --
            ├─Sequential: 1-4                        [-1, 512, 60, 60]         --
            |    └─Conv2d: 2-13                      [-1, 512, 61, 61]         1,180,160
            |    └─BatchNorm2d: 2-14                 [-1, 512, 61, 61]         1,024
            |    └─ReLU: 2-15                        [-1, 512, 61, 61]         --
            |    └─MaxPool2d: 2-16                   [-1, 512, 60, 60]         --
            ├─AdaptiveAvgPool2d: 1-5                 [-1, 512, 1, 1]           --
            ├─Flatten: 1-6                           [-1, 512]                 --
            ├─Linear: 1-7                            [-1, 128]                 65,664
            ├─ReLU: 1-8                              [-1, 128]                 --
            ├─Dropout: 1-9                           [-1, 128]                 --
            ├─Linear: 1-10                           [-1, 64]                  8,256
            ├─ReLU: 1-11                             [-1, 64]                  --
            ├─Dropout: 1-12                          [-1, 64]                  --
            ├─Linear: 1-13                           [-1, 10]                  650
            ==========================================================================================
            Total params: 1,627,466
            Trainable params: 1,627,466
            Non-trainable params: 0
            Total mult-adds (G): 5.82
            ==========================================================================================
            Input size (MB): 0.05
            Forward/backward pass size (MB): 55.84
            Params size (MB): 6.21
            Estimated Total Size (MB): 62.09
            ==========================================================================================


11. Results
   ![image](https://github.com/MuhammedM294/EuroSat/assets/89984604/af5c175d-28ba-4aaa-a5d1-919997790746)


                    
