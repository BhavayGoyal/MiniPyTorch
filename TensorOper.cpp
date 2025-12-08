// This file will define all the mathematical operators and functions needed
// TODO - IMPLIMENT BROADCASTING SO THAT IT CAN WORK DIRECTLY WITH A LOT OF STUFFS

#include "Tensor.h"
// numerical operators (+, -, *, matmul)

Tensor Tensor::operator+(Tensor &other) {
    if ((*this).shape != other.shape) throw runtime_error("Shape mismatch between operator");
    Tensor result((*this).shape, 0); vector<int> indices;
    auto add = [&](const float a, const float b) {return a + b;};
    binary_op_recursive(0, indices, other, result, add);
    return result;
}

Tensor Tensor::operator-(Tensor &other) {
    if ((*this).shape != other.shape) throw runtime_error("Shape mismatch between operator");
    Tensor result((*this).shape, 0); vector<int> indices;
    auto sub = [&](const float a, const float b) {return a - b;};
    binary_op_recursive(0, indices, other, result, sub);
    return result;
}

Tensor Tensor::operator*(Tensor &other) {
    if ((*this).shape != other.shape) throw runtime_error("Shape mismatch between operator");
    Tensor result((*this).shape, 0); vector<int> indices;
    auto mul = [&](const float a, const float b) {return a * b;};
    binary_op_recursive(0, indices, other, result, mul);
    return result;
}

Tensor Tensor::operator+(int val) {
    Tensor other((*this).shape, val);
    if ((*this).shape != other.shape) throw runtime_error("Shape mismatch between operator");
    Tensor result((*this).shape, 0); vector<int> indices;
    auto add = [&](const float a, const float b) {return a + b;};
    binary_op_recursive(0, indices, other, result, add);
    return result;
}

Tensor Tensor::operator-(int val) {
    Tensor other((*this).shape, val);
    if ((*this).shape != other.shape) throw runtime_error("Shape mismatch between operator");
    Tensor result((*this).shape, 0); vector<int> indices;
    auto sub = [&](const float a, const float b) {return a - b;};
    binary_op_recursive(0, indices, other, result, sub);
    return result;
}

Tensor Tensor::operator*(int val) {
    Tensor other((*this).shape, val);
    if ((*this).shape != other.shape) throw runtime_error("Shape mismatch between operator");
    Tensor result((*this).shape, 0); vector<int> indices;
    auto mul = [&](const float a, const float b) {return a * b;};
    binary_op_recursive(0, indices, other, result, mul);
    return result;
}

Tensor Tensor::matMul(Tensor &other) {
    if (shape.size() != 2 || other.shape.size() != 2) throw runtime_error("The matrix multiplication only supports 2D tensors....");
    int N = shape[0], K = shape[1], other_K = other.shape[0], M = other.shape[1];
    if (K != other_K) throw runtime_error("Dimension mismatch for MatMul: Columns of A must match Rows of B.");

    Tensor result({N, M},  0);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += (*this)({i, k})*other({k, j});
            }
            result({i, j}) = sum;
        }
    }

    return result;
}

void Tensor::binary_op_recursive(int dim, vector<int> &indices, Tensor &other, Tensor &result, function<float(float, float)> op) {
    if (dim == shape.size()) {
        float a = (*this)(indices);
        float b = (other)(indices);
        result(indices) = op(a, b);
        return;
    }
    for (int i = 0; i < shape[dim]; i++) {
        indices.push_back(i);
        binary_op_recursive(dim+1, indices, other, result, op);
        indices.pop_back();
    }
}