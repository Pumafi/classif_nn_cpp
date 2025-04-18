# pragma once

# include <vector>
# include <random>
# include <cmath>
# include <algorithm>

std::vector<float> vector_fill_init(int input_dim, int output_dim, float value_fill){
    std::vector<float> output(output_dim, value_fill);

    return output;
}

std::vector<std::vector<float>> matrix_2d_fill_init(int input_dim, int output_dim, float value_fill){
    std::vector<std::vector<float>> output(input_dim);
    std::transform(output.begin(), output.end(), output.begin(), [&](std::vector<float>){return vector_fill_init(input_dim, output_dim, value_fill); });
    return output;
}

float glorot_uniform_values(int input_dim, int output_dim){
    std::random_device rd;
    std::mt19937 gen(rd());

    float lower_limit = -std::sqrt(6) / std::sqrt(input_dim + output_dim);
    float higher_limit = std::sqrt(6) / std::sqrt(input_dim + output_dim);

    std::uniform_real_distribution<float> distribution(lower_limit, higher_limit);

    return distribution(gen);
}

std::vector<float> vector_glorot_uniform_init(int input_dim, int output_dim){
    std::vector<float> output(output_dim);
    std::transform(output.begin(), output.end(), output.begin(), [&](float){return glorot_uniform_values(input_dim, output_dim); });

    return output;
}

std::vector<std::vector<float>> matrix_2d_glorot_uniform_init(int input_dim, int output_dim){
    std::vector<std::vector<float>> output(input_dim);
    std::transform(output.begin(), output.end(), output.begin(), [&](std::vector<float>){return vector_glorot_uniform_init(input_dim, output_dim); });
    return output;
}