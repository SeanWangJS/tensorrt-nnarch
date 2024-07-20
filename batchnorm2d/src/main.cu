// https://www.nvidia.cn/content/dam/en-zz/zh_cn/assets/webinars/oct16/Gary_TensorRT_GTCChina2019.pdf

#include <iostream>
#include <string>
#include <fstream>

#include <NvInfer.h>

using namespace nvinfer1;

class Logger : public nvinfer1::ILogger{
    
    void log(Severity severity, const char* msg) noexcept override {
        std::cout << msg << std::endl;
    }
    
};


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


void saveEngineData(nvinfer1::IHostMemory* engine, const std::string& filepath) {

    std::ofstream engineFile(filepath, std::ios::binary);
    if(!engineFile) {
        std::cerr << "Error opening file" << std::endl;
        return;
    }   

    engineFile.write(static_cast<const char*>(engine -> data()), engine -> size());

    engineFile.close();

}

int main(int argc, char** argv) {

    Logger logger;
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    int flag = 1U << static_cast<int>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder -> createNetworkV2(flag);

    int32_t numChannels = 64;
    nvinfer1::ITensor* input = network -> addInput("input", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{4, {1, numChannels, 100, 100}});

    std::string weightDir = "../weights";
    
    nvinfer1::Weights alpha = loadWeight(weightDir, "weight");
    nvinfer1::Weights beta = loadWeight(weightDir, "bias");
    nvinfer1::Weights mean = loadWeight(weightDir, "running_mean");
    nvinfer1::Weights var = loadWeight(weightDir, "running_var");

    const float* a = static_cast<const float*>(alpha.values);
    const float* b = static_cast<const float*>(beta.values);
    const float* m = static_cast<const float*>(mean.values);
    const float* v = static_cast<const float*>(var.values);

    float* scale = new float[numChannels];
    float* shift = new float[numChannels];
    float* power = new float[numChannels];

    for(int i = 0; i < numChannels; i++) {
        scale[i] = a[i] / sqrt(v[i] + 1e-5);
        shift[i] = - a[i] / sqrt(v[i] + 1e-5) * m[i] + b[i];
        power[i] = 1.0;
    }

    nvinfer1::Weights scaleWeights;
    scaleWeights.type = nvinfer1::DataType::kFLOAT;
    scaleWeights.values = scale;
    scaleWeights.count = numChannels;

    nvinfer1::Weights shiftWeights;
    shiftWeights.type = nvinfer1::DataType::kFLOAT;
    shiftWeights.values = shift;
    shiftWeights.count = numChannels;

    nvinfer1::Weights powerWeights;
    powerWeights.type = nvinfer1::DataType::kFLOAT;
    powerWeights.values = power;
    powerWeights.count = numChannels;

    nvinfer1::IScaleLayer* bnLayer = network -> addScaleNd(*input, 
                                                           nvinfer1::ScaleMode::kCHANNEL, 
                                                           shiftWeights, 
                                                           scaleWeights, 
                                                           powerWeights, 
                                                           1);

    bnLayer->getOutput(0)->setName("output");

    network -> markOutput(*bnLayer->getOutput(0));

    nvinfer1::IBuilderConfig* config = builder -> createBuilderConfig();
    config -> setMaxWorkspaceSize(1 << 20);
    nvinfer1::IHostMemory* engine = builder -> buildSerializedNetwork(*network, *config);

    std::string savePath = "../model.engine";

    saveEngineData(engine, savePath);

    return 0;

    
}