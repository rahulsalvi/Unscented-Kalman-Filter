#include "../include/ukf.h"

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
