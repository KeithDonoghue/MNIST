#include "IdxFile.h"

#include <fstream>
#include <iostream>

void byteReverse(rInt& obj)
{
  uint8_t temp = obj.data[3];
  obj.data[3] = obj.data[0];
  obj.data[0] = temp;

  temp = obj.data[2];
  obj.data[2] = obj.data[1];
  obj.data[1] = temp;
}

std::ostream& operator<<(std::ostream& os, rInt rhs)
{
  for (int i = 0; i < INT_SIZE; ++i)
    {
      os << +rhs.data[i];
    }
  return os;
}





IdxFile::IdxFile(std::string fileName):
  mFileName(fileName),
  mFile(fileName),
  mDataRead(false)
{


  if(mFile.is_open())
    {
      std::cout << mFileName << std::endl;
      std::cout << "Open!" << std::endl;

      mFile.read((char*)&mMagic.data, 4);
  mFile.read((char*)&mNumImages.data, 4);

  mRows.value = 1;
  mCols.value = 1;

  if(mMagic.data[3] > 1)
    {
      mFile.read((char*)&mRows.data, 4);
      byteReverse(mRows);
    }

  if(mMagic.data[3] > 2)
    {
      mFile.read((char*)&mCols.data, 4);
      byteReverse(mCols);
    }


  byteReverse(mNumImages);

    }
  else
    {
      std::cout << "Not open!" << std::endl;
    }
}


void IdxFile::readData()
{
  if(!mDataRead)
    {
      if(mMagic.data[3] == 3)
	readImages();
      
      if(mMagic.data[3] == 1)
	readLabels();
    }

  mDataRead = true;
}


void IdxFile::readImages()
{
  for(int i = 0 ; i < mNumImages.value ; ++i)
    {
  std::vector<uint8_t> image;
  image.resize(mRows.value*mCols.value);
  mFile.read((char*)image.data(), image.size());
  mImages.push_back(image);
    }
}

void IdxFile::readLabels()
{
}



void IdxFile::writeImage(int index)
{
  int col = 1;
  for(auto val: mImages[index])
    {
      std::cout << +val;
      if(col++ == mCols.value)
	{
	  col = 1;
	  std::cout << std::endl;
	}
    }
}


std::vector<arma::mat> IdxFile::readImageMats()
{
  readData();
  
  std::vector<arma::mat> matrices;
  for(auto image : mImages)
    {
      std::vector<double> tempVec(image.begin(), image.end());
      arma::mat temp(tempVec.data(), mCols.value*mRows.value, 1);
      matrices.push_back(temp);
    }
  
  return matrices;
}



