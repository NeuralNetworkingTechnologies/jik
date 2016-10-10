#!/usr/bin/env bash

WGET=wget
if ! type "$WGET" &> /dev/null; then
  echo Command $WGET not found
  exit 1
fi
GUNZIP=gunzip
if ! type "$GUNZIP" &> /dev/null; then
  echo Command $GUNZIP not found
  exit 1
fi
MNIST_DIR=$(dirname "$0")/mnist
rm -rf $MNIST_DIR
mkdir -p $MNIST_DIR
$WGET -P $MNIST_DIR http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
$WGET -P $MNIST_DIR http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
$WGET -P $MNIST_DIR http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
$WGET -P $MNIST_DIR http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
$GUNZIP $MNIST_DIR/*.gz
