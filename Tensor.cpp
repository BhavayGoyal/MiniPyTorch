// This file defines the following functions - 
// Tensor Constructor
// Indexing
// Slicing
// Transpose
// Print

#include "Tensor.h"

Range::Range(int _start, int _stop, int _step) {start = _start, stop = _stop, step = _step;}

Tensor::Tensor(const vector<int> &_shape, float def_val) { // Constructor Function
    shape = _shape;
    int sh = 1; for (const auto &i : _shape) sh *= i; 
    data = make_shared <vector<float>> (sh, def_val);
    strides = vector<int> (shape.size(), 1);
    for (int i = shape.size()-2; i >= 0; i--) strides[i] = strides[i+1]*shape[i+1];
    offset = 0; // offset will be 0 only
}

float& Tensor::operator()(const vector<int> &index) { return (*data)[get_index(index)]; } // operator overloading

void Tensor::print(int precision) {
    vector<int> current_indices;
    print_recursive(0, current_indices, precision);
    cout << endl;
}

Tensor::Tensor(shared_ptr<vector<float>> _data, vector<int> _shape, vector<int> _strides, int _offset) { // straight-forward constructor
    data = _data;
    shape = _shape;
    strides = _strides;
    offset = _offset;
}

Tensor Tensor::slice (const vector<Range> &ranges) {
    vector<int> new_shape;
    vector<int> new_strides;
    int new_offset = offset;

    for (int i = 0; i < shape.size(); i++) {
        Range r = (i < ranges.size()) ? ranges[i] : Range(0, shape[i], 1);
        int start = r.start, stop = (r.stop == -1 ? shape[i] : r.stop), step = r.step;
        
        // safety checks
        if (start < 0 || start > shape[i]) throw runtime_error("Slice start out of bounds");
        if (stop < start || stop > shape[i]) throw runtime_error("Slice stop out of bounds or invalid");
        
        int newDim = (stop - start + (step - 1))/step; // ceil of (stop-start)/step
        new_shape.push_back(newDim);
        new_strides.push_back(strides[i]*step);
        new_offset += start*strides[i];
    }
    return Tensor(data, new_shape, new_strides, new_offset);
}

Tensor Tensor::T () { // transpose
    vector<int> new_shape = shape; reverse(new_shape.begin(), new_shape.end());
    vector<int> new_strides = strides; reverse(new_strides.begin(), new_strides.end());
    return Tensor(data, new_shape, new_strides, offset);
}

void Tensor::print_recursive(int dim, vector<int> &current_indices, int precision) {
    if (dim == shape.size()) return;
    cout << "[";
    for (int i = 0; i < shape[dim]; i++) {
        current_indices.push_back(i);
        // if (dim == shape.size() - 1) cout << fixed << setprecision(precision) << (*data)[get_index(current_indices)];
        if (dim == shape.size() - 1) cout << fixed << setprecision(precision) << (*this)(current_indices);
        else print_recursive(dim + 1, current_indices, precision);
        current_indices.pop_back();

        if (i < shape[dim] - 1) {
            if (dim == shape.size() - 1) {
                cout << ", "; // Separator between values in a row
            } else {
                cout << ",";
                int newlines = shape.size() - 1 - dim; // number of line breaks to give
                for(int n=0; n < newlines; n++) cout << "\n";
                cout << string(dim + 1, ' ');
            }
        }
    }
    cout << "]";
}

int Tensor::get_index(const vector<int> &index) { // Get index helper function
    if (index.size() != shape.size()) throw runtime_error("Index dimension mismatch.");
    int linear_index = offset;
    for (int i = 0; i < index.size(); i++) {
        if (index[i] < 0 || index[i] >= shape[i]) throw runtime_error("Index out of bounds.");
        linear_index += index[i]*strides[i];
    }
    if (linear_index >= data->size()) throw runtime_error("Linear index out of bounds");
    return linear_index;
}