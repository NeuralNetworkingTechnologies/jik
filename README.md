# jik

Release 0.1 (October 2016)

## Info

This project is implementing basic algorithms for two of the main deep
learning models:
* *Feed-forward Neural Networks* (FFNN) (including *Convolutional Neural
  Network* (CNN) models)
* *Recurrent Neural Networks* (RNN) (including *Long Short-Term Memory* (LSTM)
  models)

It is currently only implemented on the CPU (mono-threaded) but a
multi-threaded version as well as a CUDA version will be coming, hopefully
soon.

I tried to keep the design of the system very simple and lightweight so it's
easy to parse and understand.
There's no dependency by default, making it easy to compile, port and run.

## Structure

* core: main library, including layers, graph, model and solver
* recurrent: RNN, including LSTM
* data: placeholder for the datasets (with some scripts to download them)
* model: pre-trained models
* sandbox: examples
  * mnist  : mnist classifier (classifying the mnist dataset)
  * cifar10: cifar10 classifier (classifying the cifar10 dataset)
  * textgen: example of RNN (or LSTM) model taking an input text and
    predicting sentences

## Requirements

You must have a Linux, macOS or Windows system with:
* some C++ compiler compatible with C++ 11 (see below)
* cmake (version 2.8 or above)

## Compiler

By default, we will use gcc/g++ to compile anything.
To use clang/clang++, just define those environment variables:
* export CC=clang    (default = gcc)
* export CXX=clang++ (default = g++)

Please note that on macOS, gcc/g++ are just symlinks to clang/clang++.

Note that you need a C++ 11 (aka C++ 0x) friendly C++ compiler.

## Compilation

This project is using cmake.
Make sure you have at least version 2.8.

From the root directory, you can run:
```sh
./build.sh
```

Which is equivalent to:
```sh
mkdir build
cd build
cmake ..
make -j8
```

## Code style (cpplint)

We're using google c++ style guide:
https://google.github.io/styleguide/cppguide.html

To make sure the code is compliant with this style guide, via cpplint, run:
```sh
make lint
```

## Data

From the root directory, you can run:
```sh
./data.sh
```
in order to get all the data needed to run the sandbox examples.

The following will be downloaded:
* MNIST dataset
* CIFAR10 dataset
* A text file with Shakespeare input

## Sandbox examples

Make sure you downloaded the data before running the sandbox examples.

Then, from the build directory, you can run the following sandbox examples.

### Linear regression

This example will try to learn a scale value using linear regression.
This example will generate a lot of value X and an equal amount of values Y so
that Y = (N + eps) * X.
The model (made of 1 scale layer) will try to learn the value of N. 

Training the scale model to learn value 3.14:
```sh
sandbox/linear_regression/linear_regression -train -scale 3.14
```


### MNIST classifier

This example will classify the MNIST dataset (see here:
http://yann.lecun.com/exdb/mnist).

Training a CNN model, without batch normalization:
```sh
sandbox/mnist/mnist -dataset ../data/mnist -train
```

Training a CNN model, without batch normalization, using a SGD solver (instead
of a RMSprop solver by default):
```sh
sandbox/mnist/mnist -dataset ../data/mnist -train -solver sgd
```

Training a CNN model, with batch normalization:
```sh
sandbox/mnist/mnist -dataset ../data/mnist -train -bn
```

Training a FC model:
```sh
sandbox/mnist/mnist -dataset ../data/mnist -train -fc
```

Testing a pretrained CNN model:
```sh
sandbox/mnist/mnist -dataset ../data/mnist -model ../model/mnist_conv.model
```

Testing a pretrained FC model:
```sh
sandbox/mnist/mnist -dataset ../data/mnist -fc -model ../model/mnist_fc.model
```

Fine-tuning a pretrained CNN model:
```sh
sandbox/mnist/mnist -dataset ../data/mnist -model ../model/mnist_conv.model -train -name mnist_finetune
```

Testing a pretrained CNN model on the synthetic (rendered) MNIST dataset:
```sh
sandbox/mnist/mnist -dataset ../data/mnist_render -model ../model/mnist_conv.model
```

You can see that the model is behaving poorly here. It has been trained on
real MNIST dataset but hasn't seen any rendered data.
Let's try to train a model on both the real and synthetic MNIST datasets
(mixed model):
```sh
sandbox/mnist/mnist -dataset ../data/mnist:../data/mnist_render -train
```

Let's now test this new mixed model (pretrained) on the real MNIST dataset,
the synthetic MNIST dataset and both at the same time:
```sh
sandbox/mnist/mnist -dataset ../data/mnist -model ../model/mnist_mix_conv.model
sandbox/mnist/mnist -dataset ../data/mnist_render -model ../model/mnist_mix_conv.model
sandbox/mnist/mnist -dataset ../data/mnist:../data/mnist_render -model ../model/mnist_mix_conv.model
```

You can see that it's now giving very accurate results on all datasets.
We did the same thing for the FC model (mixed model):
```sh
sandbox/mnist/mnist -dataset ../data/mnist -fc -model ../model/mnist_mix_fc.model
sandbox/mnist/mnist -dataset ../data/mnist_render -fc -model ../model/mnist_mix_fc.model
sandbox/mnist/mnist -dataset ../data/mnist:../data/mnist_render -fc -model ../model/mnist_mix_fc.model
```

### CIFAR10 classifier

This example will classify the CIFAR10 dataset (see here:
http://www.cs.toronto.edu/~kriz/cifar.html).

Training a model, without batch normalization:
```sh
sandbox/cifar10/cifar10 -dataset ../data/cifar10 -train
```

Training a model, without batch normalization, using a SGD solver (instead of
a RMSprop solver by default):
```sh
sandbox/cifar10/cifar10 -dataset ../data/cifar10 -train -solver sgd
```

Training a model, with batch normalization:
```sh
sandbox/cifar10/cifar10 -dataset ../data/cifar10 -train -bn
```

Training a model, grayscaling the input images (from RGB):
```sh
sandbox/cifar10/cifar10 -dataset ../data/cifar10 -train -gray
```

Testing a pretrained model:
```sh
sandbox/cifar10/cifar10 -dataset ../data/cifar10 -model ../model/cifar10.model
```

Fine-tuning a pretrained model:
```sh
sandbox/cifar10/cifar10 -dataset ../data/cifar10 -model ../model/cifar10.model -train -name cifar10_finetune
```

### Text generator

This example will take an input text file and start generating sentences with
the same style using a RNN or LSTM.
Feel free to use your own text file.

Training a RNN model:
```sh
sandbox/textgen/textgen -dataset ../data/textgen/shakespeare_input.txt -model rnn
```

Training a LSTM model:
```sh
sandbox/textgen/textgen -dataset ../data/textgen/shakespeare_input.txt -model lstm
```
