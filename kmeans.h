#include "Data.h"
using namespace Eigen;

template<int K, int Dimension>
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
	int Test(Data& train_data) {
		// init centroids
		// init centroid_ids
		return 0;
	}

private:

	Matrix<double, K, 1> centroid_ids;

	void randomInitCentroids(Data& test_data) {
		// for each i from 1 to K
		//		set centroid i to be equal to a random example from the test set
	}

	void findIdsOfClosestCentroids(Data& test_data) {

	}

	void computeCentroids(Data& test_data) {

	}
};
