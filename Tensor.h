#ifndef TENSOR_H
#define TENSOR_H

#include <bits/stdc++.h>
using namespace std;

struct Range {
    int start, stop, step;
    
    // Default constructor with default values
    Range(int _start = 0, int _stop = -1, int _step = 1);
};

class Tensor {
public:
    // Member variables
    shared_ptr<vector<float>> data;
    vector<int> shape;
    vector<int> strides;
    int offset;

    // Default constructor
    Tensor(const vector<int> &_shape = {1}, float def_val = 0.0f);

    // Constructor for Tensor from pre-existing data
    Tensor(shared_ptr<vector<float>> _data, vector<int> _shape, vector<int> _strides, int _offset);

    // Accessor operator for indices
    float& operator()(const vector<int>& index);

    // Print tensor with specified precision
    void print(int precision = 2);

    // Slice operation for tensor
    Tensor slice(const vector<Range>& ranges);

    // Transpose operation for tensor
    Tensor T();

    // common operators
    Tensor operator+(Tensor &other);
    Tensor operator-(Tensor &other);
    Tensor operator*(Tensor &other);
    Tensor operator+(int other);
    Tensor operator-(int other);
    Tensor operator*(int other);
    Tensor matMul(Tensor &other);

private:
    // Recursive function to print the tensor
    void print_recursive(int dim, vector<int>& current_indices, int precision);

    // Helper function to get the linear index for a given multidimensional index
    int get_index(const vector<int>& index);

    // Helper function to impliment +, -, *
    void binary_op_recursive(int dim, vector<int> &indices, Tensor &other, Tensor &result, function<float(float, float)> op);
};

#endif // TENSOR_H
