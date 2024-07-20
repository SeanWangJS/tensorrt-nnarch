# BatchNorm Engine Implemented by TensorRT Network API

Since TensorRT not provide the BatchNorm layer directly, this sample demonstrates how to implement a BatchNorm using the TensorRT `IScaleLayer`. 

In TensorRT, the scala layer is defined as:

$$
X_{out}[:, i] = (X_{in}[:, i] \times scale[i] + shift[i])^{power[i]}
$$

Where $i$ is the channel index. And the batch norm definition is:

$$
X_{out}[:, i] = \frac{X_{in}[:, i] - mean[i]}{\sqrt{var[i] + \epsilon}} \cdot gamma[i] + beta[i]
$$

It can tansform to

$$
X_{out}[:, i] = \frac{X_{in}[:, i] \cdot gamma[i]}{\sqrt{var[i] + \epsilon}} + (beta[i] - \frac{mean[i] \cdot gamma[i]}{\sqrt{var[i] + \epsilon}})
$$

which means that 

$$
\begin{aligned}
scale[i] &= \frac{gamma[i]}{\sqrt{var[i] + \epsilon}}\\
shift[i] &= beta[i] - \frac{mean[i] \cdot gamma[i]}{\sqrt{var[i] + \epsilon}}\\
power[i] &= 1
\end{aligned}
$$

## Usage

> warning: This code is only tested on Ubuntu 20.04 platform

1. Compile the code

```shell
mkdir build && cd build
cmake .. -DTensorRT_ROOT=/usr/local/tensorrt ## you may need to change the TensorRT_ROOT
make
```

2. Save pytorch resnet18's bn1 layer weights to folder

```
python convert_weights.py
```

3. Build the batchnorm engine

```
./batchnorm2d
```
