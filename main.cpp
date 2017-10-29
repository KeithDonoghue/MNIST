#include <iostream>
#include "armadillo"
#include <vector>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <type_traits>


void vectorStuff();
void loadMNIST();


int main()
{
  loadMNIST();
  vectorStuff();
  
}

void loadMNIST()
{

  std::ifstream testImages("/home/ubuntu/code/Data/t10k-images.idx3-ubyte");
  std::ifstream testLabels("/home/ubuntu/code/Data/t10k-labels.idx1-ubyte");
  std::ifstream trainImages("/home/ubuntu/code/Data/train-images.idx3-ubyte");
  std::ifstream trainLabels("/home/ubuntu/code/Data/train-labels.idx1-ubyte");

  if(testImages.is_open())
    {
      int x,y, z, w;
      testImages >> x >> y >> z >> w;
      std::cout << x << ' ' << y  << ' ' << z << ' ' <<  w << std::endl;
      std::cout << "Open!" << std::endl;
    }
  if(testLabels.is_open())
    {
      int x,y, z, w;
      testLabels >> x >> y >> z >> w;
      std::cout << x << ' ' << y  << ' ' << z << ' ' <<  w << std::endl;
      std::cout << "Open!" << std::endl;
    }
  if(trainImages.is_open())
    {
      int x,y, z, w;
      trainImages >> x >> y >> z >> w;
      std::cout << x << ' ' << y  << ' ' << z << ' ' <<  w << std::endl;
      std::cout << "Open!" << std::endl;
    }
  if(trainLabels.is_open())
    {
      int x,y, z, w;
      trainLabels >> x >> y >> z >> w;
      std::cout << x << ' ' << y  << ' ' << z << ' ' <<  w << std::endl;
      std::cout << "Open!" << std::endl;
    }

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
