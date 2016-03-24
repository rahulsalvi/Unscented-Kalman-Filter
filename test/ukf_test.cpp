#include <gtest/gtest.h>

#include "../include/ukf.h"

class UnscentedKalmanFilterTester : public ::testing::Test {

	UnscentedKalmanFilterTester()  
	{
	
	}


	VectorXd _zeroVector;
	MatrixXd _zeroMatrix;
	VectorXd _state;
	UnscentedKalmanFilter *_filter;
};
