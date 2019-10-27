#include "Core.h"
#include "NN.h"


int main(){
    // auto nn = std::make_shared<Net>(3);
    auto x = torch::rand({1,9}, optd);
    // cout << nn->forward(x) << endl;
    FEM fem(3);
    fem.nn_init();
    cout << fem.nn_predict(x) << endl;

    // auto b = torch::rand({5,1}, optd);
    // auto A = torch::rand({5,5}, optd);
    // auto z = torch::solve(b, A);
    // cout << get<0>(z) << endl;
    // cout << torch::matmul(A, get<0>(z)) - b << endl;

    // auto A = torch::rand({5,5});
    // auto Z = torch::ones({5});
    // A[1] = Z;
    // A[2].fill_(0.0);
    // Z.fill_(0.0);
    // cout << A << endl;

    int n = 5;
    auto A = torch::rand({n,5});
    // cout << A.transpose(0,1) << endl;
    // cout << torch::detach(A) << endl;
    cout << A.clone() << endl;


    cout << torch::cat({A,A,A},0) << endl;
    cout << torch::le(torch::norm(A),10.0)<<endl;
    return 1;
}