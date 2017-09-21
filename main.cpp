#include "mnist_loader.h"
#include <png++/png.hpp>

int main() {
    // Load data
    Data train_data;
    Data test_data;
    load_data(train_data, test_data);
    
    // Init kmeans

    // Run kmeans

    // Ouput centroids as png pictures

    return 0;
}
