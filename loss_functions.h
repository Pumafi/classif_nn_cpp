# pragma once

# include <vector>
# include <algorithm>
# include <cmath>
# include <stdexcept>

class LossFunction{
    public:
        virtual float call(const std::vector<float> y_true, const std::vector<float> y_pred) = 0;
        std::vector<float> get_loss_gradient();
    protected:
        std::vector<float> loss_gradient;
};

std::vector<float> LossFunction::get_loss_gradient(){
    return loss_gradient;
}

class BinaryCrossEntropyLoss : LossFunction{
    public:
        float call(const std::vector<float> y_true, const std::vector<float> y_pred);
};

float BinaryCrossEntropyLoss::call(const std::vector<float> y_true, const std::vector<float> y_pred){
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("y_true and y_pred must be the same size.");
    }

    const float epsilon = 1e-12f;
    float loss = .0;
    loss_gradient.clear();

    for (size_t i = 0; i < y_true.size(); ++i) {
        float y = y_true[i];
        float p = std::clamp(y_pred[i], epsilon, .0f - epsilon);

        loss += -y * std::log(p) - (1. - y) * std::log(1. - p);

        float grad = -(y / p) + ((1. - y) / (1 - p));
        loss_gradient.push_back(grad);
    }

    loss /= y_true.size();

    return loss;
}