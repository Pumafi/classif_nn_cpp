# pragma once

# include <algorithm>
# include <memory>
# include <string>
# include <stdexcept>
# include "layers.h"
# include "weights_init.h"
# include "linear_algebra.h"
# include "optimizers.h"

class FullyConnectedLayer : public WeightedLayer{
    public:
        //std::vector<int> input_dim;
        //std::vector<int> output_dim;

        FullyConnectedLayer(int, int);
        FullyConnectedLayer(int, int, bool);
        FullyConnectedLayer(int, int, std::string);
        FullyConnectedLayer(int, int, bool, std::string);

        std::vector<std::vector<float>> call(const std::vector<std::vector<float>>);
        std::vector<std::vector<float>> apply_gradients(const std::vector<std::vector<float>>, std::unique_ptr<Optimizer> & optimizer);

    protected:
        std::vector<float> apply_weights(const std::vector<float>);
};

FullyConnectedLayer::FullyConnectedLayer(int input_dim_, int output_dim_){
    input_dim = input_dim_;
    output_dim = output_dim_;
    std::vector<std::vector<float>> weights_init_values(matrix_2d_glorot_uniform_init(input_dim, output_dim));
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
    std::transform(weights_init_values.begin(), weights_init_values.end(), std::back_insert_iterator(weights), [](std::vector<float> x){return x; }) ;

    use_bias = use_bias_;

    if (use_bias){
        std::vector<float> bias_init_values(vector_glorot_uniform_init(input_dim, output_dim));
        std::transform(bias_init_values.begin(), bias_init_values.end(), std::back_insert_iterator(bias), [](float x){return x; });
    }
    activation = activation_from_str(activation_name);
}

std::vector<std::vector<float>> FullyConnectedLayer::apply_gradients(const std::vector<std::vector<float>> gradient_signal, std::unique_ptr<Optimizer> & optimizer){
    
    std::vector<std::vector<float>> mean_w_gradients(weights.size(), std::vector<float>(weights[0].size(), 0.));
    std::vector<float> mean_b_gradients(bias.size(), 0.);

    std::vector<std::vector<float>> grad_in;

    std::vector<std::vector<float>> input_gradients;

    for (int b = 0; b < gradient_signal.size(); ++b){
        std::vector<float> g = element_wise_vector_multiplication(activation_gradients[b], gradient_signal[b]);

        std::vector<std::vector<float>> grad_w(outer_product(weights_gradients[b], g));

        // compute input gradient
        // compute weight gradient: outet_product(gradient_signal,)
        // compute bias gradient: gradient_signal
        
        mean_w_gradients = matrix_addition(mean_w_gradients, grad_w); // make a matrix addition in linear algebra

        if (use_bias){
            mean_b_gradients = vector_addition(mean_b_gradients, gradient_signal[b]);
        }

        //std::vector<float> grad_in(vector_matrix_multiplication(g, matrix_transpose(weights)));
        grad_in.push_back(vector_matrix_multiplication(g, matrix_transpose(weights)));
    }

    int batch_size = gradient_signal.size();

    std::transform(mean_w_gradients.begin(), mean_w_gradients.end(), mean_w_gradients.begin(), [&](std::vector<float> v){return vector_scalar_multiplication(1./batch_size, v);});
    mean_b_gradients = vector_scalar_multiplication(1./batch_size, mean_b_gradients);

    weights = optimizer->apply_gradient(weights, mean_w_gradients);

    if (use_bias){
        bias = optimizer->apply_gradient(bias, mean_b_gradients);
    }

    return grad_in;
}

std::vector<float> FullyConnectedLayer::apply_weights(const std::vector<float> input){
    if (input.size() != input_dim){
        throw std::invalid_argument("FullyConnected: invalid shape for multiplication");
    }
    std::vector<float> input_mult_w(vector_matrix_multiplication(input, weights));
    
    std::vector<float> after_bias = use_bias ? vector_addition(input_mult_w, bias) : input_mult_w;

    // compute gradients
    weights_gradients.push_back(input);
    std::vector<float>  output(call_activation(after_bias));
    activation_gradients.push_back(activation->get_gradients());

    return output;
}

std::vector<std::vector<float>> FullyConnectedLayer::call(const std::vector<std::vector<float>> input){
    std::vector<std::vector<float>> output(input.size());
    weights_gradients.clear();
    activation_gradients.clear();
    // weights_gradients = input;
    for (int b = 0; b < input.size(); ++b){
        output[b] = apply_weights(input[b]);
    }
    return output;
}

