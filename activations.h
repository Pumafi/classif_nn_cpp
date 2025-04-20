# pragma once

# include <iostream>
# include <vector>
# include <cmath>
# include <algorithm>
# include <numeric>
# include <memory>
# include <string>
#include <stdexcept>

class Activation{
    public:
        virtual std::vector<float> call(std::vector<float>) = 0;
        std::vector<float> get_gradients();
    protected:
        std::vector<float> gradients;
};

std::vector<float> Activation::get_gradients(){
    std::vector<float> output(gradients);
    return output;
}

class IdentityActivation : public Activation{
    /*
    Identity activation. Returns the input.
    */  
   public:
    std::vector<float> call(std::vector<float>);   
};

std::vector<float> IdentityActivation::call(std::vector<float> input){
    std::vector<float> output(input.begin(), input.end());

    gradients.clear();
    std::transform(input.begin(), input.end(), std::back_inserter(gradients), [](float x){return 1; });    

    return output;
}

class LogisticActivation : public Activation{   
    /*
    Logistic activation function, also called sigmoid function.
    Simple implementation that applies the function element wise to the input vector and compute gradient.
    */  
    public:
        std::vector<float> call(std::vector<float>);   
};

std::vector<float> LogisticActivation::call(std::vector<float> input){
    /*
    Apply the logistic activation element wise to the input vector.
    */
    std::vector<float> output;

    gradients.clear();
    // Compute gradient in the forward pass (TODO option to not do that)
    std::transform(input.begin(), input.end(), std::back_inserter(gradients), [](float x){return (1 / (1 + std::exp(-x))) * (1 - 1 / (1 + std::exp(-x))); });

    // Compute output
    std::transform(input.begin(), input.end(), std::back_inserter(output), [](float x){return 1 / (1 + std::exp(-x)); });

    return output;
}

class ReLU : public Activation{   
    /*
    Logistic activation function, also called sigmoid function.
    Simple implementation that applies the function element wise to the input vector and compute gradient.
    */  
    public:
        std::vector<float> call(std::vector<float>);   
};

std::vector<float> ReLU::call(std::vector<float> input){
    /*
    Apply the rectified linear unit activation element wise to the input vector.
    */
    std::vector<float> output;

    gradients.clear();
    // Compute gradient in the forward pass (TODO option to not do that)
    std::transform(input.begin(), input.end(), std::back_inserter(gradients), [](float x){return x > 0 ? 1. : 0.; });

    // Compute output
    std::transform(input.begin(), input.end(), std::back_inserter(output), [](float x){return x > 0 ? x : 0.; });

    return output;
}

class SoftmaxActivation : public Activation{
    /*
    Softmax activation function.
    */
    public:
        std::vector<float> call(std::vector<float>);   
};

std::vector<float> SoftmaxActivation::call(std::vector<float> input){

    // we substract the max value in the input vector for computational stability
    float max_input_val = *std::max_element(input.begin(), input.end());

    std::vector<float> exp_input;

    std::transform(input.begin(), input.end(), std::back_inserter(exp_input), [max_input_val](float x){return std::exp(x - max_input_val); });

    float sum_exp_input = std::accumulate(exp_input.begin(), exp_input.end(), 0.);

    std::vector<float> output;

    std::transform(exp_input.begin(), exp_input.end(), std::back_inserter(output), [sum_exp_input](float x){return x / sum_exp_input; });

    gradients.clear();
    // I am not really computing the gradient, our gradient will be zero everywhere and we will use the cross entropy, but this will give the right size
    std::transform(exp_input.begin(), exp_input.end(), std::back_inserter(gradients), [](float x){return 1; });

    return output;
}

std::unique_ptr<Activation> activation_from_str(std::string name){
    if (name == "identity"){
        return std::make_unique<IdentityActivation>();
    }
    if (name == "softmax"){
        return std::make_unique<SoftmaxActivation>();
    }
    if (name == "relu" || name == "ReLU"){
        return std::make_unique<ReLU>();
    }
    if (name == "sigmoid" || name == "logistic"){
        return std::make_unique<LogisticActivation>();
    }
    throw std::invalid_argument("unknown activation function name");
}