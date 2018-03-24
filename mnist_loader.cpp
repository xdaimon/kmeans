#include <iostream>
using std::cout; using std::endl;
#include <iomanip>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <vector>
using std::vector;

#include <eigen3/Eigen/Eigen>
using namespace Eigen;

#include "mnist_loader.h"

/* clang-format off
TRAINING SET IMAGE FILE (train-images.idx3-ubyte):

[offset] [type]          [value]          [description] 
0000     32 bit integer  0x00000803(2051) magic number 
0004     32 bit integer  60000            number of images 
0008     32 bit integer  28               number of rows 
0012     32 bit integer  28               number of columns 
0016     unsigned byte   ??               pixel 
0017     unsigned byte   ??               pixel 
........ 
xxxx     unsigned byte   ??               pixel
Pixels are organized row-wise. Pixel values range from 0 to 255. 0 -> white, 255 -> black.

TRAINING SET LABEL FILE (train-labels.idx1-ubyte):

[offset] [type]          [value]          [description] 
0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
0004     32 bit integer  60000            number of items 
0008     unsigned byte   ??               label 
0009     unsigned byte   ??               label 
........ 
xxxx     unsigned byte   ??               label
The labels values range from 0 to 9
 */ // clang-format on

void load_data(Data& test_data) {
	const char* test_img_file = "t10k-images.idx3-ubyte";
	const char* test_lbl_file = "t10k-labels.idx1-ubyte";

	// train_data gets 60000 images from the first img file
	// test_data gets all images from the second img file

    vector<unsigned char> file;
	auto read_file = [&file](const char* path, int offset) {
		std::ifstream fi(path, std::ios::binary);
		if (!fi.is_open()) {
			cout << "MNIST data file was not found" << endl;
			exit(1);
		}
        fi.seekg(offset, std::ios::beg);
		fi >> std::noskipws;
        std::istream_iterator<unsigned char> start(fi), end;
		file.assign(start, end);
    };

	// Test Images
	read_file(test_img_file, 16);
	test_data.examples = MatrixXd::Zero(28 * 28, 10000);
	for (int i = 0; i < 10000; ++i)
        for (int j = 0; j < 28; ++j)
            for (int k = 0; k < 28; ++k)
                test_data.examples(28*j+k, i) = file[j+28*k + i * 28 * 28] / 256.;

	// Test labels
	read_file(test_lbl_file, 8);
	test_data.labels = VectorXi::Zero(10000);
	for (int i = 0; i < 10000; ++i)
		test_data.labels(i) = file[i];
}
