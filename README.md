# Neural Differentiation in Deep Networks: A Theoretical Framework for Expressivity and Representational Diversity
This is the official implementation for Neural Differentiation in Deep Networks: A Theoretical Framework for Expressivity and Representational Diversity.

# Overview
We begin by developing a mathematical framework of neural differentiation, formulated at the level of individual neurons. This framework formalizes the principle that each neuron should acquire a distinct representational role within the network, thereby avoiding redundancy and maximizing collective expressivity. Differentiation is quantified through the Neural Differentiation Index (NDI), a lossaware measure that characterizes neuron significance from geometric, informational, and curvature-based perspectives within a unified framework. The NDI enables a rigorous characterization of how strongly a neuron diverges from its peers in both function and importance, and supports theoretical guarantees: we establish formal bounds on the error increase under NDI-guided elimination, thereby providing provable safety margins for network compression. Building on this foundation, we introduce Neural Differentiation Pruning (NDP) as a practical instantiation. NDP leverages NDI to perform adaptive, training-time neuron sparsification, followed by targeted fine-tuning, guiding networks toward compact yet highly differentiated backbones. Although the terminology draws loose intuition from biological differentiation, the framework is fully mathematical and architecture-agnostic. Experiments on modern vision benchmarks and architectures show that NDP achieves substantial structured sparsity while maintaining—or even improving—accuracy and robustness, underscoring the practical impact of the differentiation framework.

# Dependencies
```shell
conda create -n myenv python=3.7
conda activate myenv
conda install -c pytorch pytorch==1.9.0 torchvision==0.10.0
pip install scipy
```

# Datasets
Please download the Imagenet Dataset. 

# ResNet18 and Leaky ReLU
All used ResNet18 and Leaky ReLU models can be downloaded from here. Please put them in ResNet18().

# Run calculate NDI for a ResNet-18 trained on CIFAR-10.
 ```shell
$ python Resnet18.py
$ python Leaky ReLU.py
```
- In Leaky ReLU.py, replace activation functions ReLU with LeakyReLU.

# Run Neural sparsity for ResNet-18 on CIFAR-10.
 ```shell
$ python Resnet18.py
$ python Leaky ReLU.py
```

 # Run Weight sparsity for ResNet-18 on CIFAR-10.
 ```shell
$ python Resnet18.py
$ python Leaky ReLU.py
```
# Run MLP-Net on MNIST.
 ```shell
$ python MLP-Net.py

# Run VGG16 on CIFAR-10.
 ```shell
$ python VGG16.py

# Run MobileNet-V2 on ImageNet.
 ```shell
$ python MobileNet-V2.py
