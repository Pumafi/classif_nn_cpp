# pragma once

# include <vector>
# include <stdexcept>
# include <iostream>

std::vector<float> vector_addition(std::vector<float> vector_a, std::vector<float> vector_b){
    if (vector_a.size() != vector_b.size()){
        throw std::invalid_argument("vectors need to be the same size for addition!");
    }
    std::vector<float> output(vector_a.size());
    for (int i = 0; i < vector_a.size(); ++i){
        output[i] = vector_a[i] + vector_b[i];
    }
    return output;
}

float vector_scalar_product(std::vector<float> vector_a, std::vector<float> vector_b){
    if (vector_a.size() != vector_b.size()){
        throw std::invalid_argument("vectors need to be the same size for scalar product!");
    }
    float output = 0;
    for (int i = 0; i < vector_a.size(); ++i){
        output += vector_a[i] * vector_b[i];
    }
    return output;
}

std::vector<float> vector_matrix_multiplication(std::vector<float> vector, std::vector<std::vector<float>> matrix){
    if (vector.size() != matrix.size()){
        std::cout << "ERROR: invalid shapes" << std::endl;
        std::cout << "ERROR: invalid shapes. Size Matrix:" << matrix.size() << "Size vector: " << vector.size() << std::endl;
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

std::vector<std::vector<float>> matrix_transpose(std::vector<std::vector<float>> matrix){
    std::vector<std::vector<float>> output(matrix[0].size(), std::vector<float>(matrix.size()));
    for (int i =0; i < matrix.size(); ++i){
        for (int j = 0; j < matrix[0].size(); ++j){
            output[j][i] = matrix[i][j];
        }
    }
    return output;
}

std::vector<std::vector<float>> outer_product(std::vector<float> vector_a, std::vector<float> vector_b) {
    std::vector<std::vector<float>> output(vector_a.size(), std::vector<float>(vector_b.size()));

    for (size_t i = 0; i < vector_a.size(); ++i) {
        for (size_t j = 0; j < vector_b.size(); ++j) {
            output[i][j] = vector_a[i] * vector_b[j];
        }
    }

    return output;
}