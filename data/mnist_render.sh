#!/usr/bin/env bash

WGET=wget
if ! type "$WGET" &> /dev/null; then
  echo Command $WGET not found
  exit 1
fi
PYTHON=python
if ! type "$PYTHON" &> /dev/null; then
  echo Command $PYTHON not found
  exit 1
fi
MNIST_RENDER_DIR=$(dirname "$0")/mnist_render
rm -rf $MNIST_RENDER_DIR
mkdir -p $MNIST_RENDER_DIR
$WGET -P $MNIST_RENDER_DIR https://raw.githubusercontent.com/oliviersoares/mnist/master/mnist_render.py
$PYTHON $MNIST_RENDER_DIR/mnist_render.py -out $MNIST_RENDER_DIR -num 60000 -seed 101 -dmax 1.0 -dataset -prefix train
$PYTHON $MNIST_RENDER_DIR/mnist_render.py -out $MNIST_RENDER_DIR -num 10000 -seed 102 -dmax 1.0 -dataset -prefix t10k
rm -f $MNIST_RENDER_DIR/mnist_render.py
