#include <torch/script.h>

#include <iostream>

int main(int argc, const char* argv[]) {

  //  std::shared_ptr<torch::jit::script::Module> module =
  //                                        torch::jit::load("model.pt");

  std::shared_ptr<torch::jit::script::Module> module =
    std::make_shared<torch::jit::script::Module>(torch::jit::load("model.pt"));

  assert(module != nullptr);

  double x[3] = { 1.0, 2.0, 3.0 }; // strains (input)


  std::vector<torch::jit::IValue> inputs;
  torch::Tensor t = torch::tensor({x[0], x[1], x[2]},
                                  torch::dtype(torch::kFloat64).requires_grad(true));
  inputs.push_back(t.view({1, 3}));
  at::Tensor output = module->forward(inputs).toTensor();


  auto a = output.accessor<double, 2>();

  std::cout <<" here 1"  << std::endl;

  double y_pred[3];
  y_pred[0] = a[0][0];  // stresses (output)
  y_pred[1] = a[0][1];
  y_pred[2] = a[0][2];

  std::cout << x[0] << " " << x[1] << " " << x[2] <<  "    " << std::endl;
  std::cout << y_pred[0] << " " << y_pred[1] << " " << y_pred[2] << std::endl;

  double tangents[9];
  
  for(int i = 0; i < 3; ++i) {
    //delete the construction of the graph, if retain_graph=true can be given to the output[0][i].backward();
    //at::Tensor output = module->forward(inputs).toTensor();
    //output[0][i].backward();

    output[0][i].backward({}, true);    
    auto b = t.grad().accessor<double, 1>();
    for(int j = 0; j < 3; ++ j) {
      tangents[3 * i + j] = b[j]; // ∂Si/∂Ej
    }
    t.grad().zero_(); 
  }

  std::cout << tangents[0] << " " << tangents[1] << " " << tangents[2] <<  "\n"
	    << tangents[3] << " " << tangents[4] << " " << tangents[5] <<  "\n"
	    << tangents[6] << " " << tangents[7] << " " << tangents[8] <<  std::endl;

}
