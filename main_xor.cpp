#include <iostream>
#include <vector>
#include "model.h"
#include "layers.h"
#include "optimizers.h"
#include "fullyconnected_layer.h"

int main() {
    FullyConnectedLayer* fc1 = new FullyConnectedLayer(2, 4, true, "sigmoid"); // input 2 features, 4 neurons
    FullyConnectedLayer* fc2 = new FullyConnectedLayer(4, 4, true, "sigmoid"); // output 1 neuron
    FullyConnectedLayer* fc3 = new FullyConnectedLayer(4, 1, false, "sigmoid"); // output 1 neuron

    std::vector<Layer*> layers = {fc1, fc2, fc3};

    SGDOptimizer optimizer(0.1, "binary_crossentropy"); // learning rate 0.1, loss by name

    Model model(layers, optimizer);

    std::vector<std::vector<float>> x_train = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };
    std::vector<std::vector<float>> y_train = {
        {0.0f},
        {1.0f},
        {1.0f},
        {0.0f}
    };

    for (int epoch = 0; epoch < 100000; ++epoch) {
        float loss = model.training_step(x_train, y_train);
        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << " - Loss: " << loss << std::endl;
        }
    }

    auto predictions = model.call(x_train);
    std::cout << "\nPredictions after training:\n";
    for (size_t i = 0; i < predictions.size(); ++i) {
        std::cout << "Input: [" << x_train[i][0] << ", " << x_train[i][1] 
                  << "] -> Predicted: " << predictions[i][0] 
                  << ", True: " << y_train[i][0] << std::endl;
    }

    for (Layer* layer : layers) {
        delete layer;
    }

    return 0;
}
