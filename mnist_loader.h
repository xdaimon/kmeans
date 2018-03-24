#pragma once

#include <eigen3/Eigen/Eigen>
struct Data
{
	Eigen::MatrixXd examples;
	Eigen::VectorXi labels;
};
void load_data(Data& test_data);
