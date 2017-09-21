#include "Data.h"
#include <random>
#include <ctime>
using namespace Eigen;

template<int K, int Dimension, int Iterations>
class K_Means {
public:
	K_Means(){};

	// The mean of each feature
	Matrix<double, Dimension, 1> mean;

	// The standard deviation of each feature
	Matrix<double, Dimension, 1> standard_deviation;

	// Rows correspond to different centroids
	// Columns correspond to different features
	Matrix<double, K, Dimension> centroids;

	// Computes mean and standard_deviation on a train set
	void Train(Data& train_data) {
		// Compute mean over all examples for each feature 
		for (int i = 0; i < mean.size(); ++i)
			mean(i) = train_data.examples.col(i).mean();
		// Compute standard deviation over all examples for each feature
		for (int i = 0; i < standard_deviation.size(); ++i)
			standard_deviation(i) = ((train_data.examples.col(i).array() - mean(i)).pow(2)).sum();
		// Divide by number of train examples
		standard_deviation /= train_data.examples.rows()-1;
	}

	// Clusters data and uses that information to predict data classes
	// returns number_true_classifications
	int Test(Data& test_data) {
		randomInitCentroids(test_data);
		for (int iter = 0; iter < Iterations; ++iter) {
			findIdsOfClosestCentroids(test_data);
			computeCentroids(test_data);
			std::cout << iter << std::endl;
		}
		return computeAccuracy(test_data);
	}

private:

	Matrix<double, K, 1> centroid_ids;
	Matrix<double, K, 1> cluster_identity;

	void randomInitCentroids(Data& test_data) {
		const int N = test_data.examples.rows();
		auto eng = std::default_random_engine(std::time(0));
		auto dist = std::uniform_int_distribution<int>(0, N-1);
		for (int i = 0; i < K; ++i)
			centroids.row(i) = test_data.examples.row(dist(eng));
	}

	void findIdsOfClosestCentroids(Data& test_data) {
		const int N = test_data.examples.rows();
		for (int i = 0; i < N; ++i) {
			double min_dist = 0.;
			for (int k = 0; k < K; ++k) {
				double dist = (test_data.examples.row(i) - centroids.row(k)).norm();
				if (dist < min_dist) {
					min_dist = dist;
					centroid_ids(i) = k;
				}
			}
		}
	}

	void computeCentroids(Data& test_data) {
		const int N = test_data.examples.rows();
		for (int k = 0; k < K; ++k) {
			centroids.row(k).setZero();
			int cluster_size = 0;
			for (int i = 0; i < N; ++i) {
				if (centroid_ids(i) == k) {
					centroids.row(k) += test_data.examples.row(i);
					cluster_size++;
				}
			}
			centroids.row(k) /= cluster_size;
		}
	}

	void findClusterIdentity(Data& test_data) {
		// What digit is most likely associated with each cluster?
		// loop through the examples in cluster k
		//    find the examples label l
		//    add one to count_vectors(k, l)
		// index of element in count vector who's value is maximum is the most
		// likely digit that the label represents

		const int N = test_data.examples.rows();
		Matrix<int, K, K> count_vectors;
		count_vectors.setZero();
		for (int i = 0; i < N; ++i) {
			count_vectors(centroid_ids(i), test_data.labels(i)) += 1;
		}
		for (int k = 0; k < K; ++k) {
			int max_count = 0;
			for (int i = 0; i < K; ++i) {
				if (count_vectors(k, i) > max_count) {
					cluster_identity(k) = i;
					max_count = count_vectors(k, i);
				}
			}
		}
	}

	double computeAccuracy(Data& test_data) {
		const int N = test_data.examples.rows();
		findClusterIdentity(test_data);
		double accuracy = 0.;
		for (int i = 0; i < N; ++i)
			accuracy += cluster_identity(centroid_ids(i)) == test_data.labels(i) ? 1 : 0;
		return accuracy / N;
	}

};
