# Regularization Techniques for Image Classification Tasks 
Repository for the class project for CSCI-GA 3033-091 Introduction to Deep Learning Systems taught by Prof. Parijat Dube in Spring 2021

## Objective
To empirically understand the effects of introducing Batch Normalization, Dropout and Adaptive Gradient Clipping to a Resnet-50 model for an image classification task.

## Solution Approach
1. Measure the accuracy, training time, and time to reach 87% accuracy using different configurations of a self-implemented Resnet 50 model
2. Determine which combination of BatchNorm layers, dropout layers and dropout probability gives us the best accuracy 
3. Test the ability of Adaptive Gradient Clipping in replacing Batch Normalization

## Data Files

``` acg_ablations_0.2.csv```: Contains the results of the ablation study for Adaptive Gradient Clipping. 

```bn_dropout_batch_size_64.csv```: Contains the effects of varying the number of batchnorm and dropout layers (and dropout probability) on the accuracy, training time and time to 87% accuracy on a Resnet50 model with a batch size of 64.

## Code Files
```Batchnorm_0.ipynb```: Contains the codes (and results) for models with no batchnorm layers, a batch size of 64, and [0,1,2,3] dropout layers with dropout probabilities of [0.2, 0.5, 0.8]. The results are a part of ```bn_dropout_batch_size_64.csv```.

```Batchnorm_3.ipynb```: Contains the codes (and results) for models with 3 batchnorm layers (normalization in all layers), a batch size of 64, and [0,1,2,3] dropout layers with dropout probabilities of [0.2, 0.5, 0.8]. The results are a part of ```bn_dropout_batch_size_64.csv```.

```Batchnorm_dropout_analysis.ipynb```: Analyzing the effects of Dropout and Batchnormalization.

```ACG_ablations.ipynb```: Code for the ablation study for clip values and batch sizes for the Adaptive Gradient Clipping model. Here, we analyze the performance gained after training a single epoch of our self-implemented ResNet50 model.
The metrics are collected for an entirely normalization free model (i.e., no batch normalization layers). We set the dropout ptobability to 0.2 (the best performing probability for BN=0). We study the effects that different clip-values and batch sizes have on models with different dropout layers.
The results are stored in ```data/acg_ablations_0.2.csv```.


## Summary of Results

### Ablation Study for Adaptive Gradient Clipping
1. Training time: Not much of a variation of training time with change in clip value, but the training time decreases when batch size is increased.
2. Accuracy: The accuracy after 1 epoch doesnt have a consistent trend across a varying number of dropout layers. That being said, for higher number of dropout layers, accuracy seems to improve with increasing batch size, and for a lower number of dropout layers, it decreases slightly.
3. These observations are in line with that of this [paper](https://arxiv.org/pdf/2102.06171.pdf). Moreover, the paper suggests using higher batch sizes to emulate the behaviour of Batch Normalization layers. Similarly, they state that although the effects of clip value may not be visible in a few epochs, it helps to stabilize models with higher batch size.

### Effect of Batch Normalization and Dropout
1. Dropout: The training time and TTA (87%) mostly increases with an increase in dropout layers. The accuracy generally depends on the dropout probability and number of batchnorm layers. The best dropout probability across the board is 0.2. When we fix the number of batchnorm layers and dropout probabilities, there are very small differences in the accuracies when we change the number of dropout layers. Using a dropout probability of 0.2 is much better than using no (or, more dropout layers). In this case, using 3 dropout layers seems to be the most beneficial.

2. Batch Normalization: It seems to be better to keep 1 or 2 batchnorm layers in the Resnet50 model. 1 Batchnorm layer consistently provides the smallest training time,  and using 2 batchnorm layers give the shortest time to achieve an 87% accuracy. Here as well, it seems like keeping a dropout probability of 0.2 gives the best result, giving very little differences in accuracies between either 1 or 2 dropout layers.

Thus, the best combination for a batch size of 64 is seen by using 2 batch norm layers, 3 dropout layers of probability 0.2.