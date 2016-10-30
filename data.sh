#!/usr/bin/env bash

MODE=$1

DATA_DIR=$(dirname "$0")/data

if [ $MODE == "clean" ] ; then
  rm -rf $DATA_DIR/{mnist,mnist_render,svhn,cifar10,textgen}
else
  $DATA_DIR/mnist.sh
  $DATA_DIR/mnist_render.sh
  $DATA_DIR/svhn.sh
  $DATA_DIR/cifar10.sh
  $DATA_DIR/textgen.sh
fi
