#!/usr/bin/env bash

command_found () {
  type $1 &> /dev/null;
}

MODE=$1
if [ -z $MODE ] ; then
  MODE="release"
fi

MAKE=make
if ! command_found $MAKE ; then
  echo Command $MAKE not found
  exit 1
fi

CMAKE=cmake
if ! command_found $CMAKE ; then
  echo Command $CMAKE not found
  exit 1
fi

BUILD_DIR=$(dirname "$0")/build
rm -rf $BUILD_DIR

if [ $MODE == "clean" ] ; then
  exit 0
fi

mkdir -p $BUILD_DIR
pushd $BUILD_DIR
if [ $MODE == "debug" ] ; then
  $CMAKE -DCMAKE_BUILD_TYPE=Debug ..
else
  $CMAKE -DCMAKE_BUILD_TYPE=Release ..
fi
$MAKE -j8
$MAKE lint
popd
