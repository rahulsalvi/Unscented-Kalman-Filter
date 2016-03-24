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
			VectorXd zeroVector;
			zeroVector.setZero(STATE_DIM);
			
			MatrixXd zeroMatrix;
			zeroMatrix.setZero(STATE_DIM, STATE_DIM);
			
			VectorXd state;
			state.setOnes(STATE_DIM);

			MatrixXd covariance;
			covariance = MatrixXd::Identity(STATE_DIM, STATE_DIM);
		
			_filter.setState(state);
			_filter.setCovariance(covariance);
			_filter.setStateTransfer(stateTransfer);
			_filter.setMeasurementTransfer(measurementTransfer);
			_filter.setProcessNoise(zeroMatrix);
			_filter.setMeasurementNoise(zeroMatrix);
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
	EXPECT_EQ(STATE_DIM, _logger.sigmaPoints().rows());
	EXPECT_EQ(2*STATE_DIM+1, _logger.sigmaPoints().cols());

	ASSERT_DOUBLE_EQ(1, _logger.lambda());
	EXPECT_DOUBLE_EQ(-1, _logger.weights()[0]);
	EXPECT_DOUBLE_EQ(1, _logger.weights()[1]);
	EXPECT_DOUBLE_EQ(0.5, _logger.weights()[2]);

	_filter.setKappa(3);
	_filter.setAlpha(0.1);
	_filter.setBeta(-1);
	_filter.initialize();

	ASSERT_DOUBLE_EQ(0.01*(STATE_DIM+3), _logger.lambda());
	ASSERT_DOUBLE_EQ((_logger.lambda()-STATE_DIM)/_logger.lambda(), _logger.weights()[0]);
	EXPECT_DOUBLE_EQ(_logger.weights()[0] - 0.01, _logger.weights()[1]);
	EXPECT_DOUBLE_EQ(1/(2*_logger.lambda()), _logger.weights()[2]);
}
