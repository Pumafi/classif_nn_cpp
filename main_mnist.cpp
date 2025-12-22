#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include "model.h"
#include "layers.h"
#include "optimizers.h"
#include "fullyconnected_layer.h"

// Function to read MNIST CSV files
void load_mnist(const std::string& filename, std::vector<std::vector<float>>& images, std::vector<std::vector<float>>& labels) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << std::endl;
        exit(1);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string item;
        std::vector<float> image(784);
        std::vector<float> label(10, 0.0f);

        // First value is the label
        std::getline(ss, item, ',');
        int label_val = std::stoi(item);
        label[label_val] = 1.0f;  // one-hot encode

        // Next 784 values are pixel values
        for (int i = 0; i < 784; ++i) {
            std::getline(ss, item, ',');
            image[i] = std::stof(item) / 255.0f; // normalize
        }

        images.push_back(image);
        labels.push_back(label);
    }
}

int main() {
    FullyConnectedLayer* fc1 = new FullyConnectedLayer(784, 128, true, "relu"); // hidden layer
    FullyConnectedLayer* fc2 = new FullyConnectedLayer(128, 64, true, "relu");  // hidden layer
    FullyConnectedLayer* fc3 = new FullyConnectedLayer(64, 10, false, "softmax"); // output 10 classes

    std::vector<Layer*> layers = {fc1, fc2, fc3};

    SGDOptimizer optimizer(0.01, "categorical_crossentropy"); // learning rate 0.01

    Model model(layers, optimizer);

    std::vector<std::vector<float>> x_train, y_train;
    std::vector<std::vector<float>> x_test, y_test;

    load_mnist("MNIST_train.txt", x_train, y_train);
    load_mnist("MNIST_test.txt", x_test, y_test);

    int epochs = 50;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float loss = model.training_step(x_train, y_train);
        if (epoch % 1 == 0) {  // print every epoch
            std::cout << "Epoch " << epoch << " - Loss: " << loss << std::endl;
        }
    }

    auto predictions = model.call(x_test);
    int correct = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        int predicted_label = std::distance(predictions[i].begin(), std::max_element(predictions[i].begin(), predictions[i].end()));
        int true_label = std::distance(y_test[i].begin(), std::max_element(y_test[i].begin(), y_test[i].end()));
        if (predicted_label == true_label) correct++;
    }

    float accuracy = static_cast<float>(correct) / predictions.size();
    std::cout << "\nTest Accuracy: " << accuracy * 100.0f << "%" << std::endl;

    for (Layer* layer : layers) {
        delete layer;
    }

    return 0;
}
