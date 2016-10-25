#!/usr/bin/env bash

CWD=$(dirname "$0")
rm -rf $CWD/build $CWD/data/{mnist,mnist_render,svhn,cifar10,textgen}
