#include "Tensor.h"

int main() {
    // int n, m, k; cin >> n >> m >> k;
    // Tensor a({n, m, k}); 
    // for (int i = 0; i < n; i++) {
    //     for (int j = 0; j < m; j++) {
    //         for (int K = 0; K < k; K++) {
    //             a({i, j, K}) = (m*k)*i + k*j + K;
    //         }
    //     }
    // }

    int n, m; cin >> n >> m;
    Tensor a({n, m});

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            a({i, j}) = m*i + j;
        }
    }

    Tensor aT = a.T();
    a.print();
    aT.print();
    a.matMul(aT).print();

    return 0;
}