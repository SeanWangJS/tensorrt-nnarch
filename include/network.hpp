#include <tuple>
#include <variant>

#include <NvInfer.h>

namespace nnarch
{

using namespace nvinfer1;

class IntPair {

private:

    std::tuple<int, int> t;


public:
    
    IntPair(std::initializer_list<int> data) {
        auto b = data.begin();
        t = std::make_tuple(*b, *(b + 1));
    }
    
    int _1() {
        return std::get<0>(t);
    }

    int _2() {
        return std::get<1>(t);
    }
};

class NetworkApi {

private:

    nvinfer1::INetworkDefinition* network;

    nvinfer1::Dims getDims2(std::variant<IntPair, int> pair) {
        if(pair.index() == 0) {
            IntPair p = std::get<IntPair>(pair);
            return nvinfer1::Dims{2, {p._1(), p._2()}};
        }
        return nvinfer1::Dims{2, {std::get<int>(pair), std::get<int>(pair)}};
    }

public:

    NetworkApi(nvinfer1::INetworkDefinition* network) : network(network) {}

    ~NetworkApi() {
        network->destroy();
    }

    nvinfer1::ILayer* linear(nvinfer1::ITensor* input,
                             int32_t outFeatures,
                             nvinfer1::Weights weight,
                             nvinfer1::Weights bias,
                             const char* name = NULL) {

        nvinfer1::Dims dims = input->getDimensions();  
        int32_t batchSize = dims.d[0];

        nvinfer1::IConstantLayer* linearWeight = network -> addConstant(nvinfer1::Dims{2, {outFeatures, dims.d[1]}}, weight);
        nvinfer1::IConstantLayer* linearBias = network -> addConstant(nvinfer1::Dims{2, {batchSize, outFeatures}}, bias);

        nvinfer1::IMatrixMultiplyLayer* linearWeightLayer = network -> addMatrixMultiply(*input, nvinfer1::MatrixOperation::kNONE, *linearWeight->getOutput(0), nvinfer1::MatrixOperation::kTRANSPOSE);
        nvinfer1::IElementWiseLayer* linearBiasLayer = network -> addElementWise(*linearWeightLayer->getOutput(0), *linearBias->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);

        if (name != NULL) {
            linearBiasLayer -> setName(name);
        }

        return linearBiasLayer;
    }

    nvinfer1::IConvolutionLayer* conv2d(nvinfer1::ITensor* input, 
                                        int32_t outChannels,
                                        std::variant<IntPair, int> kernelSize,
                                        nvinfer1::Weights weight,
                                        nvinfer1::Weights bias,
                                        std::variant<IntPair, int> stride = 1,
                                        std::variant<IntPair, int> padding = 0,
                                        const char* name = NULL) {

        nvinfer1::IConvolutionLayer* layer = network->addConvolutionNd(*input, outChannels, getDims2(kernelSize), weight, bias);
        layer->setStrideNd(getDims2(stride));
        layer->setPaddingNd(getDims2(padding));

        if (name != NULL) {
            layer->setName(name);
        }

        return layer;

    }

    nvinfer1::IScaleLayer* batchNorm2d(nvinfer1::ITensor* input,
                                              int numChannels,
                                              nvinfer1::Weights weight,
                                              nvinfer1::Weights bias,
                                              nvinfer1::Weights mean,
                                              nvinfer1::Weights var,
                                              const char* name = NULL) {
        const float* a = static_cast<const float*>(weight.values);
        const float* b = static_cast<const float*>(bias.values);
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
        if (name != NULL) {
            bnLayer -> setName(name);
        }

        return bnLayer;
    }

    nvinfer1::IActivationLayer* relu(nvinfer1::ITensor* input, const char* name = NULL) {

        nvinfer1::IActivationLayer* layer = network -> addActivation(*input, nvinfer1::ActivationType::kRELU);
        if (name != NULL) {
            layer -> setName(name);
        }
        return layer;
    }

    nvinfer1::IPoolingLayer* maxpool2d(nvinfer1::ITensor* input, 
                                       std::variant<IntPair, int> kernelSize,
                                       std::variant<IntPair, int> stride = 1,
                                       std::variant<IntPair, int> padding = 0,
                                       const char* name = NULL) {

        nvinfer1::IPoolingLayer* layer = network -> addPoolingNd(*input, nvinfer1::PoolingType::kMAX, getDims2(kernelSize));
        layer -> setStrideNd(getDims2(stride));
        layer -> setPaddingNd(getDims2(padding));

        if (name != NULL) {
            layer -> setName(name);
        }

        return layer;
    }   

    nvinfer1::IShuffleLayer* reshape(nvinfer1::ITensor* input, 
                                     nvinfer1::Dims shape,
                                     const char* name = NULL) {

        nvinfer1::IShuffleLayer* layer = network -> addShuffle(*input);                                        
        layer -> setReshapeDimensions(shape);

        if (name != NULL) {
            layer -> setName(name);
        }

        return layer;
    }

    nvinfer1::IPoolingLayer* avgpool2d(nvinfer1::ITensor* input, 
                                      std::variant<IntPair, int> kernelSize,
                                      std::variant<IntPair, int> stride = 1,
                                      std::variant<IntPair, int> padding = 0,
                                      const char* name = NULL) {

        nvinfer1::IPoolingLayer* layer = network -> addPoolingNd(*input, nvinfer1::PoolingType::kAVERAGE, getDims2(kernelSize));
        layer -> setStrideNd(getDims2(stride));
        layer -> setPaddingNd(getDims2(padding));

        if (name != NULL) {
            layer -> setName(name);
        }

        return layer;
    }

};


} // namespace nn


