#!/usr/bin/env bash

DATA_DIR=$(dirname "$0")/data
$DATA_DIR/mnist.sh
$DATA_DIR/mnist_render.sh
$DATA_DIR/cifar10.sh
$DATA_DIR/textgen.sh
