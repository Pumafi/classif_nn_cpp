# pragma once

# include <vector>
#include <stdexcept>

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