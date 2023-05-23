# EuroSat
## Project Overview
In this project, a deep Convolutional Neural Network (CNNs) is built with PyTorch to classifiy Land use and cover dataset from Sentinel-2 satellite images.


![](https://raw.githubusercontent.com/phelber/EuroSAT/master/eurosat_overview_small.jpg)

## Dataset
[EuroSAT: Land Use and Land Cover Classification with Sentinel-2](https://github.com/phelber/eurosat)

This dataset consists out of 10 classes with in total 27,000 labeled and geo-referenced images. It has two versions: the RBG which includes the optical R, G and B frequency bands encoded as JPEG images and the multi-spectral version , which includes all 13 Sentinel-2 bands in the original value range.

1. [RGB](https://madm.dfki.de/files/sentinel/EuroSAT.zip) (**The employed one in this project**)
2. [Multi-spectral](https://madm.dfki.de/files/sentinel/EuroSATallBands.zip)
