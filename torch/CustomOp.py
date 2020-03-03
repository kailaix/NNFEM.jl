"""
Demo for Torch custom operators

Stiffmat:
sigma = NN(eps) * eps
here NN outputs a 3 by 3 matrix

CholOrth:
sigma = (NN(eps)*NN(eps)^T) * eps
here NN outputs a 3 by 3 lower triangular matrix with pattern
L0
L1 L2
0   0  L3
"""

import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim



def Data_Gen(name, ndata, dim=3):
    inputs = np.random.rand(ndata, dim)
    outputs = np.random.rand(ndata, dim)

    if name == "Net_Stiffmat":
        weights = np.reshape(range(dim*dim*dim), (dim*dim, dim))
        bias = np.array(range(dim*dim))

        H = np.dot(inputs, weights.T) + np.tile(bias, (ndata, 1))

        outputs = np.squeeze(np.matmul(np.reshape(H, (-1, dim, dim)), np.reshape(inputs, (ndata, dim, 1))))


    elif name == "Net_CholOrth":

        weights = np.reshape(range(4*dim), (4, dim))
        bias = np.array(range(4))

        L = np.dot(inputs, weights.T) + np.tile(bias, (ndata,1))

        H = np.zeros((ndata, dim*dim))

        H[:, 0] = L[:, 0] * L[:, 0]
        H[:, 1] = L[:, 0] * L[:, 1]
        H[:, 3] = L[:, 0] * L[:, 1]
        H[:, 4] = L[:, 1] * L[:, 1] + L[:, 2] * L[:, 2]
        H[:, 8] = L[:, 3] * L[:, 3]

        outputs = np.squeeze(np.matmul(np.reshape(H, (-1, dim, dim)), np.reshape(inputs, (ndata, dim, 1))))

    torch_inputs = torch.from_numpy(inputs).double()
    torch_outputs = torch.from_numpy(outputs).double()

    return torch_inputs, torch_outputs

class Net_Stiffmat(torch.nn.Module):
    def __init__(self):
        super(Net_Stiffmat, self).__init__()
        # self.fc1 = torch.nn.Linear(3, 10).double()
        # self.fc2 = torch.nn.Linear(10, 10).double()
        # self.fc3 = torch.nn.Linear(10, 9).double()
        self.fc3 = torch.nn.Linear(3, 9).double()


    def forward(self, x):
        # x = F.tanh(self.fc1(x))
        # x = F.tanh(self.fc2(x))
        # x = self.fc3(x)

        x = self.fc3(x)

        return x



class Net_CholOrth(torch.nn.Module):
    # L0
    # L1 L2
    # 0   0  L3
    def __init__(self):
        super(Net_CholOrth, self).__init__()
        # self.fc1 = torch.nn.Linear(3, 10).double()
        # self.fc2 = torch.nn.Linear(10, 10).double()
        # self.fc3 = torch.nn.Linear(10, 4).double()
        self.fc3 = torch.nn.Linear(3, 4).double()

    def forward(self, x):
        # x = F.tanh(self.fc1(x))
        # x = F.tanh(self.fc2(x))
        # x = self.fc3(x)
        x = self.fc3(x)

        nx = len(x)
        zeros = torch.zeros(nx, 1).double()

        y = torch.cat((x[:, 0:1] * x[:,0:1], x[:,0:1]*x[:,1:2], zeros,
                       x[:, 0:1] * x[:, 1:2], x[:,1:2]*x[:,1:2] + x[:,2:3]*x[:,2:3], zeros,
                       zeros, zeros, x[:,3:4]*x[:,3:4]), 1)
        return y




if __name__ == "__main__":

    # name = "Net_Stiffmat"
    # H = Net_Stiffmat()
    name = "Net_CholOrth"
    model = Net_CholOrth()


    ndata, dim = 1000, 3
    inputs, outputs = Data_Gen(name, ndata, dim)
    optimizer = optim.LBFGS(model.parameters(), lr=0.8, max_iter=1000, line_search_fn='strong_wolfe')

    Nite = 200
    for i in range(Nite):
        print("Iteration : ", i)
        def closure():
            optimizer.zero_grad()
            sigma = torch.squeeze(torch.bmm(torch.reshape(model(inputs), (-1, dim , dim )), torch.reshape(inputs, (ndata, dim , 1))))
            loss = torch.sum((sigma - outputs) ** 2)
            loss.backward()
            print("loss is = ", loss.item())
            return loss
        optimizer.step(closure)

    for param in model.parameters():
        print(param.data)


    example = torch.rand([1, dim]).double()
    traced_script_module = torch.jit.trace(model, example)
    output = traced_script_module(torch.ones([1, dim]).double())
    traced_script_module.save("model.pt")