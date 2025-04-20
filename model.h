# pragma once

# include <vector>
# include <memory>
# include <stdexcept>
#include <unistd.h>

# include "layers.h"
# include "optimizers.h"

class Model{
    public:
        //Model(std::vector<Layer*>, std::unique_ptr<Optimizer>);
        Model(std::vector<Layer*> layers, const SGDOptimizer& optimizer_);

        std::vector<std::vector<float>> call(std::vector<std::vector<float>>);
        float training_step(std::vector<std::vector<float>>, std::vector<std::vector<float>>);
        void fit(std::vector<std::vector<float>>, std::vector<std::vector<float>>, int);
        void fit(std::vector<std::vector<float>> x_train, std::vector<std::vector<float>> y_train, int epochs, int batch_size);

        std::unique_ptr<Optimizer> optimizer;

    protected:
        void backpropagation();
        float compute_loss(const std::vector<std::vector<float>> y_true, const std::vector<std::vector<float>> y_pred);

        
        std::vector<Layer*> layers_list;

        std::vector<std::vector<float>> loss_gradient;
};


// This does not work yet
//Model::Model(std::vector<Layer*> layers, std::unique_ptr<Optimizer> opt)
//    : layers_list(std::move(layers)), optimizer(std::move(opt)) {}

Model::Model(std::vector<Layer*> layers, const SGDOptimizer& optimizer_) {
    layers_list = layers;
    optimizer = std::make_unique<SGDOptimizer>(optimizer_);
    optimizer->loss_function = optimizer_.loss_function->clone();
}

float Model::compute_loss(const std::vector<std::vector<float>> y_true, const std::vector<std::vector<float>> y_pred){
    float loss = 0.;
    loss_gradient.clear();
    for (int b = 0; b < y_true.size(); ++b){
        if (!optimizer->loss_function){
            throw std::logic_error("Loss function undefined");
        }
        loss += optimizer->loss_function->call(y_true[b], y_pred[b]);
        loss_gradient.push_back(optimizer->loss_function->get_loss_gradient());
    }
    loss /= y_pred.size();
    return loss;
}

void Model::backpropagation(){
    if (loss_gradient.empty()){
        throw std::logic_error("Calling function backpropagation before the gradient is initialized.");
    }

    std::vector<std::vector<float>> current_layer_gradient(loss_gradient);

    for (std::vector<Layer*>::reverse_iterator riter = layers_list.rbegin(); riter != layers_list.rend(); ++riter) 
    { 
        current_layer_gradient = (*riter)->apply_gradients(current_layer_gradient, optimizer);
    } 
}

std::vector<std::vector<float>> Model::call(std::vector<std::vector<float>> inputs) {
    std::vector<std::vector<float>> outputs = inputs;

    for (Layer* layer : layers_list) {
        outputs = layer->call(outputs);
    }

    return outputs;
}

float Model::training_step(std::vector<std::vector<float>> x_batch, std::vector<std::vector<float>> y_batch) {
    std::vector<std::vector<float>> predictions = call(x_batch);
    float loss = compute_loss(y_batch, predictions);
    backpropagation();
    return loss;
}

void Model::fit(std::vector<std::vector<float>> x_train, std::vector<std::vector<float>> y_train, int epochs) {
    if (x_train.size() != y_train.size()) {
        throw std::invalid_argument("Size of x_train and y_train must match.");
    }

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float loss = training_step(x_train, y_train);
        std::cout << "\r epoch: " << epoch << ", loss: " << loss;;
    }
    std::cout << std::endl;
}

void Model::fit(std::vector<std::vector<float>> x_train, std::vector<std::vector<float>> y_train, int epochs, int batch_size) {
    if (x_train.size() != y_train.size()) {
        throw std::invalid_argument("Size of x_train and y_train must match.");
    }

    const int num_samples = x_train.size();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int i = 0; i < num_samples; i += batch_size) {
            int end = std::min(i + batch_size, num_samples);

            std::vector<std::vector<float>> x_batch(x_train.begin() + i, x_train.begin() + end);
            std::vector<std::vector<float>> y_batch(y_train.begin() + i, y_train.begin() + end);

            training_step(x_batch, y_batch);
        }
    }
}