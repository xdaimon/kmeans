#include "Data.h"
#include <random>
#include <ctime>
using namespace Eigen;

template<int K, int Dimension, int Iterations>
class K_Means {
public:
	K_Means(){};

	// Rows correspond to different centroids
	// Columns correspond to different features
	Matrix<double, K, Dimension> centroids;

	// Clusters data and uses that information to predict data classes
	// returns percent correctly classified
	double Test(Data& data) {
		centroid_ids = VectorXi::Zero(data.labels.size());
		randomInitCentroids(data);
		for (int iter = 0; iter < Iterations; ++iter) {
			findIdsOfClosestCentroids(data);
			computeCentroids(data);
			std::cout << "Iter: " << iter << "    Accuracy: " << computeAccuracy(data) << std::endl;
		}
		return computeAccuracy(data);
	}

private:

	VectorXi centroid_ids;
	Matrix<int, K, 1> cluster_identity;

	void randomInitCentroids(Data& data) {
		const int N = data.examples.rows();
		auto eng = std::default_random_engine(std::time(0));
		auto dist = std::uniform_int_distribution<int>(0, N-1);
		Matrix<int, K, 1> bool_vector;
		bool_vector.setZero();
		for (int i = 0; i < K; ++i) {
			int c = dist(eng);
			while (bool_vector(data.labels(c)))
				c = dist(eng);
			bool_vector(data.labels(c)) = 1;
			centroids.row(i) = data.examples.row(c);
		}
	}

	void findIdsOfClosestCentroids(Data& data) {
		const int N = data.examples.rows();
		for (int i = 0; i < N; ++i) {
			double min_dist = std::numeric_limits<double>::infinity();
			for (int k = 0; k < K; ++k) {
				double dist = (data.examples.row(i) - centroids.row(k)).squaredNorm();
				if (dist < min_dist) {
					min_dist = dist;
					centroid_ids(i) = k;
				}
			}
		}
	}

	void computeCentroids(Data& data) {
		const int N = data.examples.rows();
		Matrix<int, K, 1> cluster_sizes;
		centroids.setZero();
		cluster_sizes.setZero();
		for (int i = 0; i < N; ++i) {
			centroids.row(centroid_ids(i)) += data.examples.row(i);
			cluster_sizes(centroid_ids(i)) += 1;
		}
		for (int k = 0; k < K; ++k)
			centroids.row(k) /= cluster_sizes(k);
	}

	void findClusterIdentity(Data& data) {
		// What digit is most likely associated with each cluster?
		// loop through the examples in cluster k
		//    l = ith example's label
		//    add one to cluster_members(k, l)
		// index of element in count vector who's value is maximum is the most
		// likely digit that the label represents

		const int N = data.examples.rows();
		Matrix<int, K, K> cluster_members;
		cluster_members.setZero();
		cluster_identity.setZero();
		Matrix<int, K, 1> bool_vector;
		bool_vector.setZero();
		for (int i = 0; i < N; ++i) {
			cluster_members(centroid_ids(i), data.labels(i)) += 1;
		}
		for (int k = 0; k < K; ++k) {
			int max_count = -1;
			for (int i = 0; i < K; ++i) {
				if (cluster_members(k, i) > max_count) {
					cluster_identity(k) = i;
					max_count = cluster_members(k, i);
				}
			}
		}
	}

	double computeAccuracy(Data& data) {
		const int N = data.examples.rows();
		findClusterIdentity(data);
		double accuracy = 0.;
		for (int i = 0; i < N; ++i)
			accuracy += cluster_identity(centroid_ids(i)) == data.labels(i) ? 1 : 0;
		return accuracy / N;
	}

};
