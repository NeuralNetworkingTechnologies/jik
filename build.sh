#!/usr/bin/env bash

MAKE=make
if ! type "$MAKE" &> /dev/null; then
  echo Command $MAKE not found
  exit 1
fi
CMAKE=cmake
if ! type "$CMAKE" &> /dev/null; then
  echo Command $CMAKE not found
  exit 1
fi
BUILD_DIR=$(dirname "$0")/build
rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR
pushd $BUILD_DIR
$CMAKE ..
$MAKE -j8
$MAKE lint
popd
