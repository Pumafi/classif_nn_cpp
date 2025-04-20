# pragma once

# include <vector>
# include <stdexcept>
# include <algorithm>
# include <iostream>

std::vector<float> vector_scalar_multiplication(const float scalar, const std::vector<float> vector_a){
    std::vector<float> output;
    std::transform(vector_a.begin(), vector_a.end(), std::back_inserter(output), [&](float x){return x * scalar; });

    return output;
}

std::vector<float>  element_wise_vector_multiplication(const std::vector<float> vector_a, const std::vector<float> vector_b){
    if (vector_a.size() != vector_b.size()){
        throw std::invalid_argument("vectors need to be the same size for scalar product!");
    }
    std::vector<float> output(vector_a.size());
    for (int i = 0; i < vector_a.size(); ++i){
        output[i] = vector_a[i] * vector_b[i];
    }
    return output;
}

std::vector<float> vector_addition(const std::vector<float> vector_a, const std::vector<float> vector_b){
    if (vector_a.size() != vector_b.size()){
        throw std::invalid_argument("vectors need to be the same size for addition!");
    }
    std::vector<float> output(vector_a.size());
    for (int i = 0; i < vector_a.size(); ++i){
        output[i] = vector_a[i] + vector_b[i];
    }
    return output;
}

float vector_scalar_product(const std::vector<float> vector_a, const std::vector<float> vector_b){
    if (vector_a.size() != vector_b.size()){
        throw std::invalid_argument("vectors need to be the same size for scalar product!");
    }
    float output = 0;
    for (int i = 0; i < vector_a.size(); ++i){
        output += vector_a[i] * vector_b[i];
    }
    return output;
}

std::vector<float> vector_matrix_multiplication(const std::vector<float> vector, const std::vector<std::vector<float>> matrix){
    if (vector.size() != matrix.size()){
        std::cerr << "ERROR: invalid shapes" << std::endl;
        std::cerr << "ERROR: invalid shapes. Size Matrix:" << matrix.size() << "Size vector: " << vector.size() << std::endl;
        throw std::invalid_argument("matrix must have same first dimensiosn as vector for scalar product!");
    }

    std::vector<float> output(matrix[0].size());

    for (int i = 0; i < matrix[0].size(); ++i) {
        for (int j = 0; j < vector.size(); ++j) {
            output[i] += vector[j] * matrix[j][i];
        }
    }

    return output;
}

std::vector<std::vector<float>> matrix_transpose(const std::vector<std::vector<float>> matrix){
    std::vector<std::vector<float>> output(matrix[0].size(), std::vector<float>(matrix.size()));
    for (int i =0; i < matrix.size(); ++i){
        for (int j = 0; j < matrix[0].size(); ++j){
            output[j][i] = matrix[i][j];
        }
    }
    return output;
}

std::vector<std::vector<float>> outer_product(const std::vector<float> vector_a, const std::vector<float> vector_b) {
    std::vector<std::vector<float>> output(vector_a.size(), std::vector<float>(vector_b.size()));

    for (size_t i = 0; i < vector_a.size(); ++i) {
        for (size_t j = 0; j < vector_b.size(); ++j) {
            output[i][j] = vector_a[i] * vector_b[j];
        }
    }

    return output;
}

std::vector<std::vector<float>> matrix_addition(const std::vector<std::vector<float>> matrix_a, const std::vector<std::vector<float>> matrix_b) {
    if (matrix_a.size() != matrix_b.size()){
        throw std::invalid_argument("Outer dimension must be the same size for mat addition!");
    }
    if (matrix_a[0].size() != matrix_b[0].size()){
        throw std::invalid_argument("Inner dimension must be the same size for mat addition!");
    }
    std::vector<std::vector<float>> output(matrix_a.size(), std::vector<float>(matrix_a[0].size()));
    for (int i = 0; i < matrix_a.size(); ++i){
        for (int j = 0; j < matrix_a[0].size(); ++j){
            output[i][j] = matrix_a[i][j] + matrix_b[i][j];
        }
    }
    return output;
}