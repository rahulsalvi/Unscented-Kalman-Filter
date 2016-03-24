#include "../include/ukf.h"

UnscentedKalmanFilter::UnscentedKalmanFilter(VectorXd initialState,
				     	     MatrixXd initialCovariance,
					     VectorXd (*stateTransfer)(VectorXd, VectorXd, double),
					     VectorXd (*measurementTransfer)(VectorXd, double),
					     MatrixXd processNoise,
					     MatrixXd measurementNoise,
					     double   dt,
					     double   kappa,
					     double   alpha,
					     double   beta) :
					     _state(initialState),
					     _covariance(initialCovariance),
					     _stateTransfer(stateTransfer),
					     _measurementTransfer(measurementTransfer),
					     _processNoise(processNoise),
					     _measurementNoise(measurementNoise),
					     _dt(dt),
					     _kappa(kappa),
					     _alpha(alpha),
					     _beta(beta),
					     _rootFinder(_state.size())
{
	initialize();
}

void UnscentedKalmanFilter::initialize() {
	fixMatrixSizes();
	calculateConstants();
}

void UnscentedKalmanFilter::fixMatrixSizes() {
#ifdef UKF_DIMENSION_CHECKING
	if (_state.size() != _covariance.rows() || _state.size() != _covariance.cols()) {
		throw std::runtime_error("dimension mismatch: state covariance");
	}
	if (_state.size() != _processNoise.rows() || _state.size() != _processNoise.cols()) {
		throw std::runtime_error("dimension mismatch: state processnoise");
	}
	if (_measurementNoise.rows() != _measurementNoise.cols()) {
		throw std::runtime_error("dimension mismatch: measurementnoise");
	}
#endif
	_sigmaPoints.resize(_state.size(), 2*_state.size()+1);
	_sigmaPointsH.resize(_measurementNoise.rows(), 2*_state.size()+1);
	_root.resize(_state.size(), _state.size());
}

void UnscentedKalmanFilter::calculateConstants() {
	_lambda                       = _alpha * _alpha * (_state.size() * _kappa);
	_weights[MEAN_WEIGHT_0]       = (_lambda - _state.size()) / _lambda;
	_weights[COVARIANCE_WEIGHT_0] = _weights[MEAN_WEIGHT_0] + 1 - (_alpha * _alpha) + _beta;
	_weights[BOTH_WEIGHT_I]       = 0.5 / _lambda;
}

void UnscentedKalmanFilter::step(VectorXd control, VectorXd measurement) {
	createSigmaPoints();
	update(control);
	predict(measurement);
}

void UnscentedKalmanFilter::createSigmaPoints() {
	_rootFinder.compute(_lambda * _covariance);
	_root = _rootFinder.matrixL();

	_sigmaPoints.col(0) = _state;
	for (int i = 1; i < _state.size()+1; i++) {
		_sigmaPoints.col(i) = _state + _root.col(i-1);
		_sigmaPoints.col(_state.size()+i) = _state + _root.col(i-1);
	}
}