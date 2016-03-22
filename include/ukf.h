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

#ifndef UKF_H
#define UKF_H

#include <eigen3/Eigen/Dense>
#include <vector>
#include <exception>

using namespace Eigen;

#define W0M  0
#define W0C  1
#define WIMC 2

class UnscentedKalmanFilter {
	public:
		UnscentedKalmanFilter(int stateDimension,
				      int measurementDimension,
				      VectorXd initialState,
				      MatrixXd initialCovariance,
				      VectorXd (*stateTransfer)(VectorXd, double),
				      VectorXd (*measurementTransfer)(VectorXd, double),
				      MatrixXd processNoise,
				      MatrixXd measurementNoise,
				      double dt,
				      double kappa,
				      double alpha = 0.001,
				      double beta = 2) :
				      _stateDimension(stateDimension),
				      _measurementDimension(measurementDimension),
				      _state(initialState),
				      _covariance(initialCovariance),
				      _stateTransfer(stateTransfer),
				      _measurementTransfer(measurementTransfer),
				      _processNoise(processNoise),
				      _measurementNoise(measurementNoise),
				      _dt(dt),
				      _kappa(kappa),
				      _alpha(alpha),
				      _beta(beta)
{
	if (_stateDimension != _state.cols()) {
		throw std::runtime_error("dimension mismatch");
	}
	if (_stateDimension != _covariance.cols() || _stateDimension != _covariance.rows()) {
		throw std::runtime_error("dimension mismatch");
	}
	if (_stateDimension != _processNoise.cols() || _stateDimension != _processNoise.rows()) {
		throw std::runtime_error("dimension mismatch");
	}
	if (_measurementDimension != _measurementNoise.cols() || _measurementDimension != _measurementNoise.rows()){
		throw std::runtime_error("dimension mismatch");
	}
	_numSigmaPoints = (2*_stateDimension)+1;
	_lambda         = _alpha * _alpha * (_stateDimension * _kappa);
	_weights[W0M]   = (_lambda - _stateDimension) / _lambda;
	_weights[W0C]   = _weights[W0M] + 1 - (_alpha * _alpha) + _beta;
	_weights[WIMC]  = 0.5 / _lambda;
}



		VectorXd getState() {return _state;}
		MatrixXd getCovariance() {return _covariance;}

		void setState(VectorXd in) {_state = in;}
		void setProcessNoise(MatrixXd in) {_processNoise = in;}
		void setMeasurementNoise(MatrixXd in) {_measurementNoise = in;}

		void step(VectorXd measurement);
	private:
		int      _stateDimension;
		int      _measurementDimension;
		int      _numSigmaPoints;

		VectorXd _state;
		MatrixXd _covariance;

		double _weights[3];	//W0M, W0C, WIMC

		VectorXd (*_stateTransfer)(VectorXd, double);
		VectorXd (*_measurementTransfer)(VectorXd, double);

		MatrixXd _processNoise;
		MatrixXd _measurementNoise;

		double   _dt;

		double   _kappa;
		double   _alpha;
		double   _beta;
		double _lambda;

		MatrixXd _sigmaPoints;
		MatrixXd _sigmaPointsF;
		MatrixXd _sigmaPointsH;

		void predict();
		void update(VectorXd measurement);
};

#endif //UKF_H
