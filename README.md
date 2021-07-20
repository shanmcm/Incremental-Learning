# Incremental Learning
Developers: Anna Di Lorenzo, Gabriele Bruno Franco, Shannon Mc Mahon.

## Overview
Despite the recent success of neural networks, their capability is frequently limited to closed
world scenarios in which it is assumed that the semantic concepts a model has to recognize is
limited to the number of classes seen during training. Several works have investigated the
scenario known as Open World Recognition (OWR) in order to break these limiting
assumptions. In the aforementioned framework, the model must be able to: i) learn new
classes incrementally; ii) recognize when classes do or do not
belong to the knowledge it already has (open set); and iii) add these classes to its
knowledge once data for these categories is provided.
In this project we implement and study the knowledge distillation strategy to
address incremental learning challenges, and subsequently incorporate rejection
capability into the models.

## Introduction
As humans, our vision systems are inherently incremental. This means that when new visual information is incorporated existing knowledge is preserved. Consider a child familiar with the objects spoon and fork. If it is sees chopsticks, it will retain this new knowledge without forgetting previously learnt objects. The same cannot be said for vision-based applications. These systems are trained in batch settings, in which all objects are known in advance. The objective moving forward is to find more flexible strategies that are able to incrementally learn new classes. In other words, as soon as new data becomes available for a class we want to be able to learn from it incrementally and avoid retraining the model from scratch, which can be computationally unfeasible. This scenario is known as class-incremental. <br/> One possible technique consists in training
classifiers from class-incremental data streams, e.g. using
stochastic gradient descent optimization. Unfortunately this approach causes a quick deterioration of the classification accuracy. This effect is known as catastrophic forgetting. In practical terms new learning may alter weights involved in representing old learning, which in turn leads to inappreciable results.<br/>
In this paper we firstly reproduce the results of ICaRL (Incremental Classifier and Representation Learning, the original paper can be found [here](https://arxiv.org/abs/1611.07725)), a proposed technique that allows the learning of new classes incrementally. Subsequently, having compared this method with other existing ones, we move to the Open World scenario in which we show how to recognize whether or not a given sample belongs to the previously acquired knowledge, and when appropriate how to add classes to the knowledge of the system as soon as data for these categories is provided. Finally, we propose a possible modification to the defined model.

## iCaRL 
IcaRL addresses Catastrophic forgetting by use of the following:
1. Augmented Loss Function: it combines a standard classification loss with a distillation loss. The first encourages improvements of the feature representation which results in good classification of new classes, while the second is a regularization term used to avoid the loss of previously learnt information;
2. Augmented Training set: consists of training data and stored Exemplars. The latter are a representative set of samples from the distribution for each seen class. Their presence ensures that at least some information of the previous classes enters the training process;
3. NME Classifier: Nearest Mean of Exemplars multi-class classifier. Each class has a prototype vector which consists of the average feature vector of the exemplars of such class. Given an unlabeled image, it is assigned the label of the class that minimises the distance between the feature vector of the image and the prototype vector of a class;

The experiments are run on the CIFAR-100 dataset, using a training of 10 by 10 class scenarios. Specifically, the dataset is randomly divided on 10 sets of 10 classes, and each set is learned on a different training step. The testing set instead includes all the classes seen in the current learning step and also the previous training steps (i.e. after step 3 we test on the 30 known classes).<br/>
We compare iCaRL with two other approaches:<br/>
- Finetuning: learns ordinary multi-class network without taking into account any measures to mitigate catastrophic forgetting. 
- Learning without forgetting: has a similar philosophy to that of iCaRL, as it makes use of an additional Distillation Loss term. However, it does not use the Exemplar Sets.
iCaRL, as expected, outperforms both methods. Its confusion matrix shows a homogeneous behaviour over all classes, both in terms of correct predictions (diagonal entries) and mistakes (non diagonal entries). This is due to the absence of intrinsic bias towards classes encountered early or late in the learning process.<br/>

## Abliation Study

iCaRL is a combination of NME Classifier and a Binary Cross Entropy (BCE) loss coupled with iCaRL distillation. With the aim  of highlighting the gap between iCaRL and other possible choices we considered different combinations of classifiers and distillation losses.<br/>
As classfiiers we chose:
- NME, as in iCaRL
- KNN, since like the former it also relies on the concept of neighbours
- SVM with RBF Kernel

while for the losses:<br/>
- BCE, as in iCaRL
- L2 losss (Least Squared Errors)

We highlight that the distillation loss is also used as classification loss. This is due to the fact that different losses can give results on different scales. Another possible solution would be to balance different classification and distillation losses by mean of a weighing function. Such approach is not however inspected in this project.

## Open World 

Currently working on this part of the project.