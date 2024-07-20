#include <stdlib.h>
#include <random>
#include <map>
#include <string>
#include <iostream>
#include <fstream>
#include <tuple>

#include <NvInfer.h>

#include "network.hpp"

using namespace nvinfer1;
using namespace nnarch;

nvinfer1::Weights generateRandomWeights(int count) {

    float* values = new float[count];

    for(int i = 0; i < count; i++) {
        float d = static_cast<float>(rand() / static_cast<float>(RAND_MAX));
        values[i] = d;
    }

    nvinfer1::Weights weights;
    weights.type = nvinfer1::DataType::kFLOAT;
    weights.values = values;
    weights.count = count;

    return weights;

}

nvinfer1::Weights loadWeight(const std::string& weightDir, const std::string& weightName) {
    std::string weightPath = weightDir + "/" + weightName + ".bin";
    std::ifstream file(weightPath, std::ios::binary|std::ios::ate);
    if(!file) {
        std::cerr << "Error opening file" << std::endl;
        return nvinfer1::Weights();
    }

    file.seekg(0, file.end);
    int length = file.tellg();
    file.seekg(0, file.beg);

    char* data = new char[length];
    file.read(data, length);

    nvinfer1::Weights weight;
    weight.type = nvinfer1::DataType::kFLOAT;
    weight.values = data;
    weight.count = length / sizeof(float);

    return weight;

}

class Logger : public nvinfer1::ILogger{
    
    void log(Severity severity, const char* msg) noexcept override {
        std::cout << msg << std::endl;
    }
    
};

void saveEngineData(nvinfer1::IHostMemory* engine, const std::string& filepath) {

    std::ofstream engineFile(filepath, std::ios::binary);
    if(!engineFile) {
        std::cerr << "Error opening file" << std::endl;
        return;
    }   

    engineFile.write(static_cast<const char*>(engine -> data()), engine -> size());

    engineFile.close();

}

void print(nvinfer1::Dims dims) {

    std::string str = "[";

    int MAX_DIMS = dims.MAX_DIMS;

    for(int i = 0; i < MAX_DIMS - 1; i++) {
        str += std::to_string(dims.d[i]) + ", ";
    }

    str += std::to_string(dims.d[MAX_DIMS - 1]) + "]";

    std::cout << str << std::endl;

}

int main(int argc, char** argv) {

    std::map<std::string, nvinfer1::Weights> weightsMap;

    int num_weights = 16;

    std::string weightNames[num_weights] = {
        "features.0.weight",
        "features.0.bias",
        "features.3.weight",
        "features.3.bias",
        "features.6.weight",
        "features.6.bias",
        "features.8.weight",
        "features.8.bias",
        "features.10.weight",
        "features.10.bias",
        "classifier.1.weight",
        "classifier.1.bias",
        "classifier.4.weight",
        "classifier.4.bias",
        "classifier.6.weight",
        "classifier.6.bias"
    };

    std::string weightDir = "../weights";
    for(int i = 0; i < num_weights; i++) {
        weightsMap[weightNames[i]] = loadWeight(weightDir, weightNames[i]);
    }

    Logger logger;
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    int flag = 1U << static_cast<int>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder -> createNetworkV2(flag);

    nnarch::NetworkApi api = nnarch::NetworkApi(network);

    nvinfer1::ITensor* input = network -> addInput("input", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{4, {1, 3, 224, 224}});

    // Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    nvinfer1::IConvolutionLayer* conv1 = api.conv2d(input, 64, 11, weightsMap["features.0.weight"], weightsMap["features.0.bias"], 4, 2, "conv1");    
    
    std::cout << "Conv1 output dimensions: " << std::endl;
    print(conv1 -> getOutput(0) -> getDimensions());

    // ReLU(inplace=True)
    nvinfer1::IActivationLayer* relu1 = api.relu(conv1 -> getOutput(0), "relu1");

    // MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    nvinfer1::IPoolingLayer* max_pool1 = api.maxpool2d(relu1 -> getOutput(0), 3, 2, 0, "max_pool1");

    std::cout << "MaxPool1 output dimensions: " << std::endl;
    print(max_pool1 -> getOutput(0) -> getDimensions());

    // Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    nvinfer1::IConvolutionLayer* conv2 = api.conv2d(max_pool1 -> getOutput(0), 192, 5, weightsMap["features.3.weight"], weightsMap["features.3.bias"], 1, 2, "conv2");

    std::cout << "Conv2 output dimensions: " << std::endl;
    print(conv2 -> getOutput(0) -> getDimensions());

    // ReLU(inplace=True)
    nvinfer1::IActivationLayer* relu2 = api.relu(conv2 -> getOutput(0), "relu2");

    // MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    nvinfer1::IPoolingLayer* max_pool2 = api.maxpool2d(relu2 -> getOutput(0), 3, 2, 0, "max_pool2");

    std::cout << "MaxPool2 output dimensions: " << std::endl;
    print(max_pool2 -> getOutput(0) -> getDimensions());

    // Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    nvinfer1::IConvolutionLayer* conv3 = api.conv2d(max_pool2 -> getOutput(0), 384, 3, weightsMap["features.6.weight"], weightsMap["features.6.bias"], 1, 1, "conv3");

    std::cout << "Conv3 output dimensions: " << std::endl;
    print(conv3 -> getOutput(0) -> getDimensions());

    // ReLU(inplace=True)
    nvinfer1::IActivationLayer* relu3 = api.relu(conv3 -> getOutput(0), "relu3");
    
    // Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    nvinfer1::IConvolutionLayer* conv4 = api.conv2d(relu3 -> getOutput(0), 256, 3, weightsMap["features.8.weight"], weightsMap["features.8.bias"], 1, 1, "conv4");

    std::cout << "Conv4 output dimensions: " << std::endl;
    print(conv4 -> getOutput(0) -> getDimensions());

    // ReLU(inplace=True)
    nvinfer1::IActivationLayer* relu4 = api.relu(conv4 -> getOutput(0), "relu4");
    
    // Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    nvinfer1::IConvolutionLayer* conv5 = api.conv2d(relu4 -> getOutput(0), 256, 3, weightsMap["features.10.weight"], weightsMap["features.10.bias"], 1, 1, "conv5");

    std::cout << "Conv5 output dimensions: " << std::endl;
    print(conv5 -> getOutput(0) -> getDimensions());

    // ReLU(inplace=True)
    nvinfer1::IActivationLayer* relu5 = api.relu(conv5 -> getOutput(0), "relu5");

    // MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    nvinfer1::IPoolingLayer* max_pool5 = api.maxpool2d(relu5 -> getOutput(0), 3, 2, 0, "max_pool5");

    std::cout << "MaxPool5 output dimensions: " << std::endl;
    print(max_pool5 -> getOutput(0) -> getDimensions());

    // AdaptiveAvgPool2d(output_size=(6, 6))
    nvinfer1::IPoolingLayer* avg_pool = api.avgpool2d(max_pool5 -> getOutput(0), 1, 1, 0, "avg_pool");

    nvinfer1::Dims dims = avg_pool -> getOutput(0) -> getDimensions();

    std::cout << "AvgPool output dimensions: " << std::endl;
    print(avg_pool -> getOutput(0) -> getDimensions());

    int32_t const batch = dims.d[0];
    int32_t const length = dims.d[1] * dims.d[2] * dims.d[3];
    nvinfer1::IShuffleLayer* reshapeLayer = api.reshape(avg_pool -> getOutput(0), Dims{2, {batch, length}}, "reshape");

    std::cout << "reshape output dimensions: " << std::endl;    
    print(reshapeLayer -> getOutput(0) -> getDimensions());

    // Linear(in_features=9216, out_features=4096, bias=True)
    nvinfer1::ILayer* linear1 = api.linear(reshapeLayer -> getOutput(0), 4096, weightsMap["classifier.1.weight"], weightsMap["classifier.1.bias"], "linear1");

    // ReLU(inplace=True)
    nvinfer1::IActivationLayer *relu6 = api.relu(linear1 -> getOutput(0), "relu6");

    // Linear(in_features=4096, out_features=4096, bias=True)
    nvinfer1::ILayer* linear2 = api.linear(relu6 -> getOutput(0), 4096, weightsMap["classifier.4.weight"], weightsMap["classifier.4.bias"], "linear2");
    
    // ReLU(inplace=True)
    nvinfer1::IActivationLayer *relu7 = api.relu(linear2 -> getOutput(0), "relu7");

    // Linear(in_features=4096, out_features=1000, bias=True)
    nvinfer1::ILayer* linear3 = api.linear(relu7 -> getOutput(0), 1000, weightsMap["classifier.6.weight"], weightsMap["classifier.6.bias"], "linear3");

    linear3 -> getOutput(0) -> setName("output");
    network -> markOutput(*linear3 -> getOutput(0));

    nvinfer1::IBuilderConfig* config = builder -> createBuilderConfig();
    config -> setMaxWorkspaceSize(1<<20);
    nvinfer1::IHostMemory* engine = builder -> buildSerializedNetwork(*network, *config);

    std::string savePath = "./model.engine";
    saveEngineData(engine, savePath);

    return 0;
}
