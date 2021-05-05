# Regularization Techniques for Image Classification Tasks 
Repository for the class project for CSCI-GA 3033-091 Introduction to Deep Learning Systems taught by Prof. Parijat Dube in Spring 2021

## Objective
To empirically understand the effects of introducing Batch Normalization, Dropout and Adaptive Gradient Clipping to a Resnet-50 model for an image classification task.

## Solution Approach
1. Measure the accuracy, training time, and time to reach 87% accuracy using different configurations of a self-implemented Resnet 50 model
2. Determine which combination of BatchNorm layers, dropout layers and dropout probability gives us the best accuracy 
3. Test the ability of Adaptive Gradient Clipping in replacing Batch Normalization

## Data Files
### Analyzed Data

``` acg_ablations_0.2.csv```: Contains the results of the ablation study for Adaptive Gradient Clipping. 

## Code Files
```Batchnorm_0.ipynb```: Contains the codes (and results) for models with no batchnorm layers, a batch size of 64, and [0,1,2,3] dropout layers with dropout probabilities of [0.2, 0.5, 0.8].

```ACG_ablations.ipynb```: Code for the ablation study for clip values and batch sizes for the Adaptive Gradient Clipping model. Here, we analyze the performance gained after training a single epoch of our self-implemented ResNet50 model.
The metrics are connected for an entirely normalization free model (i.e., no batch normalizationlayers). We set the dropout ptobability to 0.2 (the best performing probability for BN=0). We study the effects that different clip-values and batch sizes have on models with different dropout layers.
The results are stored in ```data/acg_ablations_0.2.csv```.


## Summary of Results

### Ablation Study for Adaptive Gradient Clipping
1. Training time: Not much of a variation of training time with change in clip value, but the training time decreases when batch size is increased.
2. Accuracy: The accuracy after 1 epoch doesnt have a consistent trend across a varying number of dropout layers. That being said, for higher number of dropout layers, accuracy seems to improve with increasing batch size, and for a lower number of dropout layers, it decreases slightly.
3. These observations are in line with that of this [paper](https://arxiv.org/pdf/2102.06171.pdf). Moreover, the paper suggests using higher batch sizes to emulate the behaviour of Batch Normalization layers. Similarly, they state that although the effects of clip value may not be visible in a few epochs, it helps to stabilize models with higher batch size.
