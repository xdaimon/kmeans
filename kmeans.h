#include <random>
#include <ctime>

#include <eigen3/Eigen/Eigen>
using namespace Eigen;

template<int Dimension, int Iterations>
class K_Means {
public:
    K_Means() = delete;
    K_Means(int num_centroids) : num_centroids(num_centroids) {
        cluster_identity = MatrixXi::Zero(num_centroids, 1);
        centroids = MatrixXd::Zero(Dimension, num_centroids);
    }

    // Each column is a centroid
	MatrixXd centroids;

	// Clusters data and uses that information to predict data labels
	// Returns percentage of the data correctly classified
	double Test(Data& data) {
		centroid_ids = VectorXi::Zero(data.labels.size());
		randomInitCentroids(data);
		for (int iter = 0; iter < Iterations; ++iter) {
			findIdsOfClosestCentroids(data);
			computeCentroids(data);
		}
		return computeAccuracy(data);
	}

private:
    int num_centroids;

	VectorXi centroid_ids; // which centroid does the ith datum belong to?
	MatrixXi cluster_identity; // classify based on most frequent label in cluster

    // Set the ith centroid to a random mnist image
	void randomInitCentroids(Data& data) {
        const int N = data.labels.size();
		auto eng = std::default_random_engine(std::time(0));
		auto dist = std::uniform_int_distribution<int>(0, N-1);
		for (int i = 0; i < num_centroids; ++i) {
            centroids.col(i) = data.examples.col(dist(eng));
		}
	}

    // For each data member x, finds the centroid that is closest to x
	void findIdsOfClosestCentroids(Data& data) {
        const int N = data.labels.size();
		for (int i = 0; i < N; ++i) {
			double min_dist = std::numeric_limits<double>::infinity();
			for (int k = 0; k < num_centroids; ++k) {
				const double dist = (data.examples.col(i) - centroids.col(k)).squaredNorm();
				if (dist < min_dist) {
					min_dist = dist;
					centroid_ids(i) = k;
				}
			}
		}
	}

    // Updates the kth centroid to be equal to the average of the vectors that are nearest to the kth centroid
	void computeCentroids(Data& data) {
        const int N = data.labels.size();
		MatrixXi cluster_sizes = MatrixXi::Zero(num_centroids, 1);
		centroids.setZero();
		for (int i = 0; i < N; ++i) {
            centroids.col(centroid_ids(i)) += data.examples.col(i);
			cluster_sizes(centroid_ids(i)) += 1;
		}
        for (int k = 0; k < num_centroids; ++k) {
			centroids.col(k) /= cluster_sizes(k);
        }
	}

    // Assigns the label of a cluster to be the label that occurs most often in the cluster's members.
	void findClusterIdentity(Data& data) {
		MatrixXi cluster_label_count = MatrixXi::Zero(num_centroids, 10);
        const int N = data.labels.size();
		for (int i = 0; i < N; ++i) {
			cluster_label_count(centroid_ids(i), data.labels(i)) += 1;
		}
		cluster_identity.setZero();
		for (int k = 0; k < num_centroids; ++k) {
            int guessed_label;
            cluster_label_count.row(k).maxCoeff(&guessed_label);
            cluster_identity(k) = guessed_label;
		}
	}

    // How homogeneous are the labels of data members in a cluster?
	double computeAccuracy(Data& data) {
        const int N = data.labels.size();
		findClusterIdentity(data);
		double accuracy = 0.;
        for (int i = 0; i < N; ++i) {
            accuracy += (cluster_identity(centroid_ids(i)) == data.labels(i)) ? 1 : 0;
        }
		return accuracy / N;
	}

};
