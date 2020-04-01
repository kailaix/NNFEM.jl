# Torch Demo

This demo introduces several basic customized nerual networks

## Custom operators:

* Mat:
```
sigma = NN(eps) * eps
here NN outputs a 3 by 3 matrix
The result is not unique, W[3k + i,j] + W[3k + j,i] = Const. k,i,j=0,1,2
```

* OrthMat:
```
sigma = NN(eps) * eps
here NN outputs a 3 by 3 orthotropic matrix with pattern
H0 H1  0
H1 H2  0
0   0  H3
```

* CholOrthMat:
```
sigma = (NN(eps)*NN(eps)^T) * eps
here NN outputs a 3 by 3 lower triangular matrix with pattern
L0
L1 L2
0   0  L3
The result is not unique due to the sign difference
```


The CustomOp.py trains and generates the model as model.pt
You can play with different Matrices, and different neural network hyperparameters.

## C++ wrapper:
You can also plug your customized nerual network model.pt to the C++ FEM code
There is an example in cpptorch.cpp

* download libtorch
```
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.2.0.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.2.0.zip
```
* modify the library files in cm, do
```
source cm
```

* compile the cpp file
```
make
```

* run the generated excutable file
```
./cpptorch
```
It will output the 1 by 3 input, 1 by 3 output, and 3 by 3 Jacobian



