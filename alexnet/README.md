# Alexnet implementation by TensorRT Network API

## Usage

> warning: This code is only tested on Ubuntu 20.04 platform

1. Compile the code

```shell
mkdir build && cd build
cmake .. -DTensorRT_ROOT=/usr/local/tensorrt ## you may need to change the TensorRT_ROOT
make
```

2. Save pytorch alexnet weights to folder

```
python convert_weights.py
```

3. Build the alexnet engine

```
./alexnet
```