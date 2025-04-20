# pragma once

# include <vector>
# include <algorithm>
# include <memory>
# include <string>

# include "linear_algebra.h"
# include "loss_functions.h"

class Optimizer{
    public:
        float get_learning_rate();
        virtual std::vector<std::vector<float>> apply_gradient(const std::vector<std::vector<float>>, const std::vector<std::vector<float>>) = 0;
        virtual std::vector<float> apply_gradient(std::vector<float>, std::vector<float>) = 0;
        
        std::unique_ptr<LossFunction> loss_function;
        
    protected:
        float learning_rate;

};

float Optimizer::get_learning_rate(){
    return learning_rate;
}

class SGDOptimizer : public Optimizer{
    public:
        // SGDOptimizer(float, LossFunction&); TODO
        SGDOptimizer(float learning_rate, std::string);
        SGDOptimizer(const SGDOptimizer&);

        std::vector<std::vector<float>> apply_gradient(const std::vector<std::vector<float>>, const std::vector<std::vector<float>>);
        std::vector<float> apply_gradient(const std::vector<float>, const std::vector<float>);
        std::unique_ptr<LossFunction> loss_function;
    protected:
        float learning_rate;
        
};

SGDOptimizer::SGDOptimizer(const SGDOptimizer& other){
    learning_rate = other.learning_rate;
    loss_function = other.loss_function->clone();
}

// SGDOptimizer::SGDOptimizer(float learning_rate, LossFunction& loss_function): std::make_unique(loss_function){} TODO

SGDOptimizer::SGDOptimizer(float learning_rate_, std::string loss_name){
    learning_rate = learning_rate_;
    loss_function = loss_function_from_str(loss_name);
}


std::vector<std::vector<float>> SGDOptimizer::apply_gradient(const std::vector<std::vector<float>> weights, const std::vector<std::vector<float>> gradients){
    // matrix weights
    std::vector<std::vector<float>>minus_lr_gradients;
    std::transform(gradients.begin(), gradients.end(), std::back_inserter(minus_lr_gradients), [&](std::vector<float> v){return vector_scalar_multiplication(-learning_rate, v);});
    std::vector<std::vector<float>>output(matrix_addition(weights, minus_lr_gradients));
    return output;
}

std::vector<float> SGDOptimizer::apply_gradient(const std::vector<float> weights, const std::vector<float> gradients){
    // vectors weights
    std::vector<float>minus_lr_gradients(vector_scalar_multiplication(-learning_rate, gradients));
    std::vector<float>output(vector_addition(weights, minus_lr_gradients));
    return output;
}
