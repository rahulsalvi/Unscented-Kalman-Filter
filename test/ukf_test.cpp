/*
The MIT License (MIT)

Copyright (c) 2016 Rahul Salvi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#include <iostream>
#include <gtest/gtest.h>
#include <eigen3/Eigen/Core>

#include "../include/ukf.h"
#include "../include/ukfLogger.h"

using namespace Eigen;

#define STATE_DIM 2

VectorXd stateTransfer(VectorXd state, VectorXd control, double dt) {
	return state;
}

VectorXd measurementTransfer(VectorXd measurement, double dt) {
	return measurement;
}

class UnscentedKalmanFilterTester : public ::testing::Test {
	public:
		UnscentedKalmanFilterTester() : _logger(_filter) {}

		virtual void SetUp() {
			_filter.setState(VectorXd::Ones(STATE_DIM,1));
			_filter.setCovariance(MatrixXd::Identity(STATE_DIM, STATE_DIM));
			_filter.setStateTransfer(stateTransfer);
			_filter.setMeasurementTransfer(measurementTransfer);
			_filter.setProcessNoise(MatrixXd::Zero(STATE_DIM, STATE_DIM));
			_filter.setMeasurementNoise(MatrixXd::Zero(STATE_DIM, STATE_DIM));
			_filter.setDt(0.01);
			_filter.setKappa(-1);
			_filter.setAlpha(1);
			_filter.setBeta(2);
			_filter.initialize();
		}

		UnscentedKalmanFilter _filter;
		UnscentedKalmanFilterLogger _logger;
};

TEST_F(UnscentedKalmanFilterTester, InitializesCorrectly) {
	EXPECT_EQ(_logger.sigmaPoints().rows(), STATE_DIM);
	EXPECT_EQ(_logger.sigmaPoints().cols(), 2*STATE_DIM+1);

	ASSERT_DOUBLE_EQ(_logger.lambda(),     1);
	EXPECT_DOUBLE_EQ(_logger.weights()[0], -1);
	EXPECT_DOUBLE_EQ(_logger.weights()[1], 1);
	EXPECT_DOUBLE_EQ(_logger.weights()[2], 0.5);

	_filter.setKappa(3);
	_filter.setAlpha(0.1);
	_filter.setBeta(-1);
	_filter.initialize();

	ASSERT_DOUBLE_EQ(_logger.lambda(),     0.01*(STATE_DIM+3));
	ASSERT_DOUBLE_EQ(_logger.weights()[0], (_logger.lambda()-STATE_DIM)/_logger.lambda());
	EXPECT_DOUBLE_EQ(_logger.weights()[1], _logger.weights()[0] - 0.01);
	EXPECT_DOUBLE_EQ(_logger.weights()[2], 1/(2*_logger.lambda()));
}

TEST_F(UnscentedKalmanFilterTester, CreatesSigmaPointsCorrectly) {
	_filter.createSigmaPoints();
	
	Matrix<double, STATE_DIM, 1> vec;

	vec << 1, 1;
	EXPECT_TRUE(_logger.sigmaPoints().col(0).isApprox(vec, 0.01));

	vec << 2, 1;
	EXPECT_TRUE(_logger.sigmaPoints().col(1).isApprox(vec, 0.01));

	vec << 1, 2;
	EXPECT_TRUE(_logger.sigmaPoints().col(2).isApprox(vec, 0.01));

	vec << 0, 1;
	EXPECT_TRUE(_logger.sigmaPoints().col(3).isApprox(vec, 0.01));

	vec << 1, 0;
	EXPECT_TRUE(_logger.sigmaPoints().col(4).isApprox(vec, 0.01));

	vec << 2.5, -2.5;
	_filter.setState(vec);

	Matrix<double, STATE_DIM, STATE_DIM> cov;
	cov << 0.25, 2, 2, 80;
	_filter.setCovariance(cov);

	_filter.createSigmaPoints();
	
	vec << 2.5, -2.5;
	EXPECT_TRUE(_logger.sigmaPoints().col(0).isApprox(vec, 0.01));
	
	vec << 3.0, 1.5;
	EXPECT_TRUE(_logger.sigmaPoints().col(1).isApprox(vec, 0.01));
	
	vec << 2.5, 5.5;
	EXPECT_TRUE(_logger.sigmaPoints().col(2).isApprox(vec, 0.01));
	
	vec << 2.0, -6.5;
	EXPECT_TRUE(_logger.sigmaPoints().col(3).isApprox(vec, 0.01));
	
	vec << 2.5, -10.5;
	EXPECT_TRUE(_logger.sigmaPoints().col(4).isApprox(vec, 0.01));
}
