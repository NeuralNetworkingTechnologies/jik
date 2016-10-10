#!/usr/bin/env bash

WGET=wget
if ! type "$WGET" &> /dev/null; then
  echo Command $WGET not found
  exit 1
fi
TAR=tar
if ! type "$TAR" &> /dev/null; then
  echo Command $TAR not found
  exit 1
fi
CIFAR10_DIR=$(dirname "$0")/cifar10
rm -rf $CIFAR10_DIR
mkdir -p $CIFAR10_DIR
$WGET -P $CIFAR10_DIR https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
$TAR xzf $CIFAR10_DIR/*.tar.gz -C $CIFAR10_DIR --strip-components=1
rm -f $CIFAR10_DIR/*.tar.gz
