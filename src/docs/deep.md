# Deep Learning

* [Home](https://supaerodatascience.github.io/deep-learning/)
* [Github repository](https://github.com/SupaeroDataScience/deep-learning/)

In this class, we construct simple deep ANNs using the PyTorch library on the Fashion MNIST example.

[Notebook](https://github.com/SupaeroDataScience/deep-learning/blob/main/deep/Deep%20Learning.ipynb)

Instead of calculating backpropagation by hand as in [the first
class](https://supaerodatascience.github.io/deep-learning/ANN.html), this class
uses automatic differentiation built in to PyTorch. This is built on the
[autograd](https://pytorch.org/docs/stable/notes/autograd.html) package which
has its own
[tutorial](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html).

One important concept is convolutional neural network layers. The Stanford
CS231N class has a good interactive
[demonstration](https://cs231n.github.io/convolutional-networks/) of
convolution. [This
page](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md) shows
demonstrations of stride, padding, and dilation.

To facilitate the use of Torch, we introduce
[ignite](https://github.com/pytorch/ignite) at the end of class. An example of
using ignite is given in [this
notebook](https://github.com/SupaeroDataScience/deep-learning/blob/main/deep/Pytorch%20Ignite.ipynb)

## Additional Resources

[The deep learning book](https://www.deeplearningbook.org/) is fully available
online and contains many great examples. Notebook versions of those examples are
available [here](https://github.com/hadrienj/deepLearningBook-Notes). [Chapter
9](https://www.deeplearningbook.org/contents/convnets.html) specifically covers
convolutional neural networks.


