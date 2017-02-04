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
SVHN_DIR=$(dirname "$0")/svhn
rm -rf $SVHN_DIR
mkdir -p $SVHN_DIR
$WGET -P $SVHN_DIR https://raw.githubusercontent.com/oliviersoares/mnist/master/svhn_to_mnist.py
$WGET -P $SVHN_DIR https://raw.githubusercontent.com/oliviersoares/mnist/master/svhn_to_cifar10.py
$WGET -P $SVHN_DIR http://ufldl.stanford.edu/housenumbers/test_32x32.mat
$PYTHON $SVHN_DIR/svhn_to_mnist.py   -svhn $SVHN_DIR/test_32x32.mat -out $SVHN_DIR -prefix t10k
$PYTHON $SVHN_DIR/svhn_to_cifar10.py -svhn $SVHN_DIR/test_32x32.mat -out $SVHN_DIR -prefix test
$WGET -P $SVHN_DIR http://ufldl.stanford.edu/housenumbers/train_32x32.mat
$PYTHON $SVHN_DIR/svhn_to_mnist.py   -svhn $SVHN_DIR/train_32x32.mat -out $SVHN_DIR -prefix train
$PYTHON $SVHN_DIR/svhn_to_cifar10.py -svhn $SVHN_DIR/train_32x32.mat -out $SVHN_DIR -prefix data
rm -f $SVHN_DIR/svhn_to_mnist.py
rm -f $SVHN_DIR/svhn_to_cifar10.py
