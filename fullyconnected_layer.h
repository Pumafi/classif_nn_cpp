# pragma once

# include <algorithm>
# include <memory>
# include <string>
# include <stdexcept>
# include "layers.h"
# include "weights_init.h"
# include "linear_algebra.h"

class FullyConnectedLayer : public WeightedLayer{
    public:
        //std::vector<int> input_dim;
        //std::vector<int> output_dim;

        FullyConnectedLayer(int, int);
        FullyConnectedLayer(int, int, bool);
        FullyConnectedLayer(int, int, std::string);
        FullyConnectedLayer(int, int, bool, std::string);

        std::vector<std::vector<float>> call(std::vector<std::vector<float>>);
        std::vector<float> apply_gradients(std::vector<float>);

    protected:
        std::vector<float> apply_weights(std::vector<float>);
};

FullyConnectedLayer::FullyConnectedLayer(int input_dim_, int output_dim_){
    input_dim = input_dim_;
    output_dim = output_dim_;
    std::vector<std::vector<float>> weights_init_values(matrix_2d_glorot_uniform_init(input_dim, output_dim));
    //std::vector<std::vector<float>> weights_init_values(matrix_2d_fill_init(input_dim, output_dim, 1.));
    std::transform(weights_init_values.begin(), weights_init_values.end(), std::back_insert_iterator(weights), [](std::vector<float> x){return x; }) ;

    use_bias = true;

    std::vector<float> bias_init_values(vector_glorot_uniform_init(input_dim, output_dim));
    std::transform(bias_init_values.begin(), bias_init_values.end(), std::back_insert_iterator(bias), [](float x){return x; });

    activation = std::make_unique<IdentityActivation>();
}

FullyConnectedLayer::FullyConnectedLayer(int input_dim_, int output_dim_, bool use_bias_){
    input_dim = input_dim_;
    output_dim = output_dim_;
    std::vector<std::vector<float>> weights_init_values(matrix_2d_glorot_uniform_init(input_dim, output_dim));
    //std::vector<std::vector<float>> weights_init_values(matrix_2d_fill_init(input_dim, output_dim, 1.));

    std::transform(weights_init_values.begin(), weights_init_values.end(), std::back_insert_iterator(weights), [](std::vector<float> x){return x; }) ;

    use_bias = use_bias_;

    if (use_bias){
        std::vector<float> bias_init_values(vector_glorot_uniform_init(input_dim, output_dim));
        std::transform(bias_init_values.begin(), bias_init_values.end(), std::back_insert_iterator(bias), [](float x){return x; });
    }
    activation = std::make_unique<IdentityActivation>();
}


FullyConnectedLayer::FullyConnectedLayer(int input_dim_, int output_dim_, std::string activation_name){
    input_dim = input_dim_;
    output_dim = output_dim_;
    std::vector<std::vector<float>> weights_init_values(matrix_2d_glorot_uniform_init(input_dim, output_dim));
    //std::vector<std::vector<float>> weights_init_values(matrix_2d_fill_init(input_dim, output_dim, 1.));
    std::transform(weights_init_values.begin(), weights_init_values.end(), std::back_insert_iterator(weights), [](std::vector<float> x){return x; }) ;

    use_bias = true;

    std::vector<float> bias_init_values(vector_glorot_uniform_init(input_dim, output_dim));
    std::transform(bias_init_values.begin(), bias_init_values.end(), std::back_insert_iterator(bias), [](float x){return x; });

    activation = activation_from_str(activation_name);
}

FullyConnectedLayer::FullyConnectedLayer(int input_dim_, int output_dim_, bool use_bias_, std::string activation_name){
    input_dim = input_dim_;
    output_dim = output_dim_;
    std::vector<std::vector<float>> weights_init_values(matrix_2d_glorot_uniform_init(input_dim, output_dim));
    //std::vector<std::vector<float>> weights_init_values(matrix_2d_fill_init(input_dim, output_dim, 1.));
    std::transform(weights_init_values.begin(), weights_init_values.end(), std::back_insert_iterator(weights), [](std::vector<float> x){return x; }) ;

    use_bias = use_bias_;

    if (use_bias){
        std::vector<float> bias_init_values(vector_glorot_uniform_init(input_dim, output_dim));
        std::transform(bias_init_values.begin(), bias_init_values.end(), std::back_insert_iterator(bias), [](float x){return x; });
    }
    activation = activation_from_str(activation_name);
}

std::vector<float> FullyConnectedLayer::apply_gradients(std::vector<float> gradient_signal){
    // return the next gradient signal
    return bias; // TODO
}

std::vector<float> FullyConnectedLayer::apply_weights(std::vector<float> input){
    if (input.size() != input_dim){
        throw std::invalid_argument("FullyConnected: invalid shape for multiplication");
    }
    std::vector<float> input_mult_w(vector_matrix_multiplication(input, weights));
    
    std::vector<float> after_bias = use_bias ? vector_addition(input_mult_w, bias) : input_mult_w;

    return call_activation(after_bias);
}

std::vector<std::vector<float>> FullyConnectedLayer::call(std::vector<std::vector<float>> input){
    std::vector<std::vector<float>> output(input.size());
    for (int b = 0; b < input.size(); ++b){
        // Not using a transform, I already deleted the function one to try to optimize ("Early optimization...")
        // I'll compute the gradient here eventually
        output[b] = apply_weights(input[b]);
    }
    return output;
}

