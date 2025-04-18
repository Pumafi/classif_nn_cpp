# pragma once

# include <vector>
# include <memory>
# include "activations.h"

class Layer{
    // Abstract Layer class
    public:
        int input_dim;
        int output_dim;

        virtual std::vector<std::vector<float>> call(std::vector<std::vector<float>> input) = 0;

        std::vector<float> get_gradients();
        std::vector<float> call_activation(std::vector<float>);

    protected:
        std::unique_ptr<Activation> activation;

        std::vector<float> gradients;
};

std::vector<float> Layer::call_activation(std::vector<float> input){
    return activation->call(input);
}

std::vector<float> Layer::get_gradients(){
    std::vector<float> output(gradients);
    return output;
}

//class MaxPooling : public Layer{
// TODO: no matrices for now
//};

//class Flatten : public Layer{
// TODO: no matrices for now
//};

class WeightedLayer : public Layer{
    public:
        std::vector<std::vector<float>> get_weights();
        std::vector<float> get_bias();

        std::vector<std::vector<float>> get_weights_gradients();
        std::vector<float> get_bias_gradients();

        virtual std::vector<float> apply_gradients(std::vector<float>) = 0;

    protected:
        virtual std::vector<float> apply_weights(std::vector<float>) = 0;

        bool use_bias;

        std::vector<std::vector<float>> weights;
        std::vector<float> bias;
        

        std::vector<std::vector<float>> weights_gradients;
        std::vector<float> bias_gradients;
};

std::vector<std::vector<float>> WeightedLayer::get_weights_gradients(){
    std::vector<std::vector<float>> output(weights_gradients);
    return output;
}

std::vector<float> WeightedLayer::get_bias_gradients(){
    if (!use_bias){
        throw std::logic_error("Cannot call \"get_bias\" if \"use_bias=False\"");
    }
    std::vector<float> output(bias_gradients);
    return output;
}

std::vector<std::vector<float>> WeightedLayer::get_weights(){
    std::vector<std::vector<float>> output(weights);
    return output;
}

std::vector<float> WeightedLayer::get_bias(){
    if (!use_bias){
        throw std::logic_error("Cannot call \"get_bias\" if \"use_bias=False\"");
    }

    std::vector<float> output(bias);
    return output;
}

//class Conv2DLayer : public WeightedLayer{
// TODO: not doing matrices for now
//};