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

#include "../include/ukf.h"

UnscentedKalmanFilter::UnscentedKalmanFilter(VectorXd initialState,
                                             MatrixXd initialCovariance,
                                             VectorXd (*stateTransfer)(VectorXd, VectorXd, double),
                                             VectorXd (*measurementTransfer)(VectorXd),
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
#ifndef LOW_MEMORY
    _sigmaPointsF.resize(_state.size(), 2*_state.size()+1);
#endif
    _sigmaPointsH.resize(_measurementNoise.rows(), 2*_state.size()+1);
    _measurementState.resize(_measurementNoise.rows());
    _measurementCovariance.resize(_measurementNoise.rows(), _measurementNoise.rows());
    _crossCovariance.resize(_state.size(), _measurementNoise.rows());
    _kalmanGain.resize(_state.size(), _measurementNoise.rows());
    _root.resize(_state.size(), _state.size());
}

void UnscentedKalmanFilter::calculateConstants() {
    _lambda                       = _alpha * _alpha * (_state.size() + _kappa);
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
        _sigmaPoints.col(_state.size()+i) = _state - _root.col(i-1);
    }
}

void UnscentedKalmanFilter::predict(VectorXd control) {
#ifdef LOW_MEMORY
    MatrixXd* transformedSigmas = &_sigmaPoints;
#else
    MatrixXd* transformedSigmas = &_sigmaPointsF;
#endif
    (*transformedSigmas).col(0) = _stateTransfer(_sigmaPoints.col(0), control, _dt);
    _state = _weights[MEAN_WEIGHT_0] * (*transformedSigmas).col(0);
    for(int i = 1; i < _sigmaPoints.cols(); i++) {
        (*transformedSigmas).col(i) = _stateTransfer(_sigmaPoints.col(i), control, _dt);
        _state += _weights[BOTH_WEIGHT_I] * (*transformedSigmas).col(i);
    }

    _covariance = _weights[COVARIANCE_WEIGHT_0] * ((*transformedSigmas).col(0) - _state) * ((*transformedSigmas).col(0) - _state).transpose();
    for(int i = 1; i < _sigmaPoints.cols(); i++) {
        _covariance += _weights[BOTH_WEIGHT_I] * ((*transformedSigmas).col(i) - _state) * ((*transformedSigmas).col(i) - _state).transpose();
    }
    _covariance += _processNoise;
}

void UnscentedKalmanFilter::update(VectorXd measurement) {
#ifdef UKF_DIMENSION_CHECKING
    if (measurement.size() != _measurementState.size()) {
        throw std::runtime_error("dimension mismatch: measurement measurementstate");
    }
#endif
#ifdef LOW_MEMORY
    MatrixXd* container = &_sigmaPoints;
#else
    MatrixXd* container = &_sigmaPointsF;
#endif
    _sigmaPointsH.col(0) = _measurementTransfer((*container).col(0));
    _measurementState = _weights[MEAN_WEIGHT_0] * _sigmaPointsH.col(0);
    for(int i = 1; i < _sigmaPoints.cols(); i++) {
        _sigmaPointsH.col(i) = _measurementTransfer((*container).col(i));
        _measurementState += _weights[BOTH_WEIGHT_I] * _sigmaPointsH.col(i);
    }

    _measurementCovariance = _weights[COVARIANCE_WEIGHT_0] * (_sigmaPointsH.col(0) - _measurementState) * (_sigmaPointsH.col(0) - _measurementState).transpose();
    _crossCovariance = _weights[COVARIANCE_WEIGHT_0] * ((*container).col(0) - _state) * (_sigmaPointsH.col(0) - _measurementState).transpose();
    for(int i = 1; i < _sigmaPoints.cols(); i++) {
        _measurementCovariance += _weights[BOTH_WEIGHT_I] * (_sigmaPointsH.col(i) - _measurementState) * (_sigmaPointsH.col(i) - _measurementState).transpose();
        _crossCovariance += _weights[BOTH_WEIGHT_I] * ((*container).col(i) - _state) * (_sigmaPointsH.col(i) - _measurementState).transpose();
    }
    _measurementCovariance += _measurementNoise;

    _kalmanGain = _crossCovariance * _measurementCovariance.inverse();

    _state += _kalmanGain * (measurement - _measurementState);
    _covariance -= _kalmanGain * _measurementCovariance * _kalmanGain.transpose();
}
