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
	png::image< png::rgb_pixel > image(28, 28);
	for (size_t y = 0; y < image.get_height(); ++y)
	{
		for (size_t x = 0; x < image.get_width(); ++x)
		{
			unsigned char p = 255 - int(255.*kmeans.centroids.block<1,D>(0,0)(x + 28*y));
			image[y][x] = png::rgb_pixel(p, p, p);
		}
	}
	image.write("rgb.png");

	return 0;
}
