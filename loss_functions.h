# pragma once

# include <vector>
# include <algorithm>
# include <cmath>
# include <stdexcept>
# include <memory>

class LossFunction{
    public:
        virtual float call(const std::vector<float> y_true, const std::vector<float> y_pred) = 0;
        std::vector<float> get_loss_gradient();
        virtual std::unique_ptr<LossFunction> clone() = 0;

    protected:
        std::vector<float> loss_gradient;
};

std::vector<float> LossFunction::get_loss_gradient(){
    return loss_gradient;
}

class BinaryCrossEntropyLoss : public LossFunction{
    public:
        BinaryCrossEntropyLoss() {};
        float call(const std::vector<float> y_true, const std::vector<float> y_pred);
        std::unique_ptr<LossFunction> clone();
};

std::unique_ptr<LossFunction> BinaryCrossEntropyLoss::clone(){
    return std::make_unique<BinaryCrossEntropyLoss>(*this);
}

float BinaryCrossEntropyLoss::call(const std::vector<float> y_true, const std::vector<float> y_pred){
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("y_true and y_pred must be the same size.");
    }

    const float epsilon = 1e-8f;
    float loss = .0;
    loss_gradient.clear();

    for (size_t i = 0; i < y_true.size(); ++i) {
        float y = y_true[i];
        float y_hat = y_pred[i];
        //std::cout << "       debug2 y " << y << std::endl;
        float p = std::min(0.999f, std::max(epsilon, y_hat));
        //std::cout << "       debug2 p " << p << std::endl;
        //std::cout << "       debug2 val " << 1.f - epsilon << std::endl;

        loss += -y * std::log(p) - (1. - y) * std::log(1. - p);

        float grad = -(y / y_hat) + ((1. - y) / (1 - y_hat));
        loss_gradient.push_back(grad);
    }

    loss /= y_true.size();

    return loss;
}

std::unique_ptr<LossFunction> loss_function_from_str(std::string name){
    if (name == "binary_crossentropy" || name == "binarycrossentropy"){
        return std::make_unique<BinaryCrossEntropyLoss>();
    }
    throw std::invalid_argument("unknown loss function function name");
}