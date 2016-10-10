#!/usr/bin/env bash

CWD=$(dirname "$0")
rm -rf $CWD/build $CWD/data/{mnist,cifar10,textgen}
