#include <iostream>
#include "armadillo"
#include "IdxFile.h"



#include <vector>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <type_traits>



void vectorStuff();
void loadMNIST();
arma::mat sigmoid(arma::mat);

int main()
{
  loadMNIST();
  // vectorStuff();
  
}





void loadMNIST()
{

  IdxFile trainImagesFile("/home/ubuntu/code/Data/train-images.idx3-ubyte");
  IdxFile trainLabelsFile("/home/ubuntu/code/Data/train-labels.idx1-ubyte");
  IdxFile testImagesFile("/home/ubuntu/code/Data/t10k-images.idx3-ubyte");
  IdxFile testLabelsFile("/home/ubuntu/code/Data/t10k-labels.idx1-ubyte");

  std::vector<arma::mat> trainMatrices = trainImagesFile.readImageMats();
    std::vector<arma::mat> trainLabels =  trainLabelsFile.readLabelMats();
   std::vector<arma::mat> testMatrices = testImagesFile.readImageMats();
  std::vector<arma::mat> testLabels = testLabelsFile.readLabelMats();

  

  /*
  std::cout << trainMatrices[1].t() << std::endl;
  std::cout << testMatrices[1].t() << std::endl;

  std::cout << trainLabels[1] << std::endl;
  std::cout << testLabels[1] << std::endl;
  */
  arma::mat Weights1(15, 784, arma::fill::randu);
  arma::mat Weights2(10, 15, arma::fill::randu);

  Weights1 =  Weights1 - 0.5;
  Weights2 =  Weights2 - 0.5;

  arma::mat Biases1(15, 1, arma::fill::randu);
  arma::mat Biases2(10, 1, arma::fill::randu);

  Biases1 = Biases1 - 0.5;
  Biases2 = Biases2 - 0.5;
  
  arma::mat Z1 = Weights1 * trainMatrices[1];
  std::cout << Z1 << std::endl;
  std::cout << Biases1 << std::endl;
  arma::mat A1 = sigmoid(Z1 + Biases1);
  std::cout << A1 << std::endl;
  std::cout << A1*10 << std::endl;
  std::cout << A1*100 << std::endl;
  std::cout << A1*1000 << std::endl;

  arma::mat Z2 = Weights2 * A1;
  arma::mat A2 = sigmoid(Z2 + Biases2);
  //  std::cout << A2 << std::endl;


  arma::mat C = A2 - trainLabels[1];
  //std::cout << C << std::endl;
}


void typeStuf()
{
  std::cout << std::is_integral<int>::value << std::endl;
  std::cout << std::is_integral<double>::value << std::endl;
  std::cout << std::is_integral<float>::value << std::endl;
  std::cout << std::is_integral<uint8_t>::value << std::endl;
  std::cout << std::is_integral<int8_t>::value << std::endl;
  std::cout << "Hello World!" << std::endl;
  
}

arma::mat sigmoid(arma::mat input)
{
    auto sigmoid = [](arma::mat::elem_type& val)
    {
      double temp = 1 + std::exp(-val);
      val = 1/temp;
    };

    return input.for_each(sigmoid);
}


void vectorStuff()
{
  auto sigmoid = [](arma::mat::elem_type& val)
    {
      double temp = 1 + std::exp(-val);
      val = 1/temp;
    };

  
  std::vector<double> weights1{2, 2, 2, 2};
  arma::mat Theta1(weights1.data(), 2, 2);

  std::vector<double> Biases1{-1, 1};
  arma::mat B1(Biases1.data(), 2, 1);

  std::vector<double> weights2{2, 2};
  arma::mat Theta2(weights2.data(), 1, 2);

  std::vector<double> Biases2{-1};
  arma::mat B2(Biases2.data(), 1, 1);
  

  std::vector<double> inputs1{0, 0};
  std::vector<double> inputs2{1, 0};
  std::vector<double> inputs3{0, 1};
  std::vector<double> inputs4{1, 1};


  arma::mat X1(inputs1.data(), 2, 1);
  arma::mat X2(inputs2.data(), 2, 1);
  arma::mat X3(inputs3.data(), 2, 1);
  arma::mat X4(inputs4.data(), 2, 1);


  std::vector<std::pair<arma::mat, int>> inputs;
  inputs.push_back(std::make_pair(X1, 0));
  inputs.push_back(std::make_pair(X2, 1));
  inputs.push_back(std::make_pair(X3, 1));
  inputs.push_back(std::make_pair(X4, 0));
  
  for(auto pair : inputs)
    {
      auto X = pair.first;
      auto y = pair.second;
      arma::mat R = Theta1*X + B1;
      R.for_each(sigmoid);

      std::cout << R << std::endl;

      arma::mat out = Theta2*R + B2;
      out.for_each(sigmoid);
      std::cout << out << std::endl;
      std::cout << out - y << std::endl;
    }

  
  /*

  
  auto sigmoid = [](arma::mat::elem_type& val)
    {
      double temp = 1 + std::exp(-val);
      val = 1/temp;
    };


  
  std::cout << CircToeplitz.for_each(sigmoid) << std::endl;

  std::cout << toeplitz%CircToeplitz << std::endl;
  std::cout << "Hello World!" << std::endl;
  */
}
