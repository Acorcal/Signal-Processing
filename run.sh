#!/usr/bin/env bash
set -e

cmake --build build
./build/hello_cpp.exe "$@"
