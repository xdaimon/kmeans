#include <png++/png.hpp>
#include <eigen3/Eigen/Eigen>
#include <thread>
#include <map>
#include <mutex>

#include "mnist_loader.h"
#include "kmeans.h"

using namespace std;

int main() {
    // Load data
    Data test_data;
    load_data(test_data);

    vector<thread> threads;
    map<int, double> accuracies;
    mutex mtx;

    // Compute kmeans with i number of centroids
    const auto processor = [&](int i) {
        const int Dimension = 28 * 28;
        const int Iters = 10;
        K_Means<Dimension, Iters> kmeans(i);

        // Run kmeans
        double acc = kmeans.Test(test_data);
        mtx.lock();
        accuracies[i] = acc;
        mtx.unlock();
    };

    for (int i = 20; i < 40; ++i) {
        threads.emplace_back(processor, i);

        // Only launch 4 threads at a time
        if ((i + 1) % 4 == 0) {
            for (auto &thread : threads) {
                thread.join();
            }
            for (auto &acc : accuracies) {
                cout << "K: " << acc.first << " accuracy: " << acc.second << endl;
            }

            threads.clear();
            accuracies.clear();
        }
    }

    /* // Save a centroid as a png
    png::image< png::rgb_pixel > image(28, 28);
    for (size_t x = 0; x < image.get_width(); ++x) {
    for (size_t y = 0; y < image.get_height(); ++y) {
    unsigned char p = 255 - int(255.*kmeans.centroids.block<Dimension, 1>(0, 9)(28*x + y));
    image[y][x] = png::rgb_pixel(p, p, p);
    }
    }
    image.write(string("rgb") + to_string(i) + string(".png"));
    */

    return 0;
}
