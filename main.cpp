#include <png++/png.hpp>

#include "mnist_loader.h"
#include "kmeans.h"
#include "Data.h"

int main() {
	// Load data
	Data train_data;
	Data test_data;
	load_data(train_data, test_data);

	// Init kmeans
	const int K = 10;
	const int D = 28*28;
	const int Iters = 20;
	K_Means<K, D, Iters> kmeans;

	// Run kmeans
	double accuracy = kmeans.Test(test_data);
	std::cout << "Accuracy: " << accuracy << std::endl;

	// Ouput centroids as png pictures

	return 0;
}
