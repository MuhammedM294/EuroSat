# EuroSat
## Project Overview
In this project, a deep Convolutional Neural Network (CNNs) is built with PyTorch to classify Land use and cover dataset from Sentinel-2 satellite images.


![](https://raw.githubusercontent.com/phelber/EuroSAT/master/eurosat_overview_small.jpg)

## Dataset
[EuroSAT: Land Use and Land Cover Classification with Sentinel-2](https://github.com/phelber/eurosat)

This dataset consists out of 10 classes with in total 27,000 labeled and geo-referenced images. It has two versions: the RBG which includes the optical R, G and B frequency bands encoded as JPEG images and the multi-spectral version , which includes all 13 Sentinel-2 bands in the original value range.

1. [RGB](https://madm.dfki.de/files/sentinel/EuroSAT.zip) (**The employed one in this project**)
2. [Multi-spectral](https://madm.dfki.de/files/sentinel/EuroSATallBands.zip)

## Training
### The First Training
The first training is performed using:
1. Model : VGG19 
2. Apply Dropout: True
3. Learning Rate : 0.001
4. Number of training epochs: 25
5. Traing Data Size: 22000
6. Test Data Size: 5000
7. Model Summary: 

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
                            
                            
9. Results 
     ![image](https://github.com/MuhammedM294/EuroSat/assets/89984604/9ad8985b-e28b-4534-811a-e17c0b098195)
                    
