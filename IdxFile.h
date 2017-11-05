#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <armadillo>

#define INT_SIZE 4


typedef union
{
  uint8_t data[INT_SIZE];
  int value;
  
} rInt;




class IdxFile{
public:
  IdxFile(std::string);
  void readData();
  void writeImage(int);
  std::vector<arma::mat> readImageMats();
  std::vector<arma::mat> readLabelMats();

private:
  void readImages();
    void readLabels();
  int getDataSize(rInt);


private:
  bool mDataRead;
  std::string mFileName;
  std::ifstream mFile;
  rInt mMagic, mNumImages, mRows, mCols;
  std::vector<uint8_t> mLabels;
  std::vector<std::vector<uint8_t>> mImages;

};

