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

#ifndef UKF_FUNCTIONS
#include "ukf.h"
#else

template<int STATE_DIM, int MEASUREMENT_DIM, int CONTROL_DIM>
UnscentedKalmanFilter<STATE_DIM, MEASUREMENT_DIM, CONTROL_DIM>::UnscentedKalmanFilter(
        stateVector       initialState,
        stateMatrix       initialCovariance,
        stateVector       (*stateTransfer)      (stateVector, controlVector, double),
        measurementVector (*measurementTransfer)(stateVector),
        stateMatrix       processNoise,
        measurementMatrix measurementNoise,
        double            dt,
        double            kappa,
        double            alpha,
        double            beta) :
        _state               (initialState),
        _covariance          (initialCovariance),
        _stateTransfer       (stateTransfer),
        _measurementTransfer (measurementTransfer),
        _processNoise        (processNoise),
        _measurementNoise    (measurementNoise),
        _dt                  (dt),
        _kappa               (kappa),
        _alpha               (alpha),
        _beta                (beta)
{
    initialize();
}

template<int STATE_DIM, int MEASUREMENT_DIM, int CONTROL_DIM>
void UnscentedKalmanFilter<STATE_DIM, MEASUREMENT_DIM, CONTROL_DIM>::initialize() {
    _lambda                       = _alpha * _alpha * (_state.size() + _kappa);
    _weights[MEAN_WEIGHT_0]       = (_lambda - _state.size()) / _lambda;
    _weights[COVARIANCE_WEIGHT_0] = _weights[MEAN_WEIGHT_0] + 1 - (_alpha * _alpha) + _beta;
    _weights[BOTH_WEIGHT_I]       = 0.5 / _lambda;
}

template<int STATE_DIM, int MEASUREMENT_DIM, int CONTROL_DIM>
Matrix<double, STATE_DIM, 1>
UnscentedKalmanFilter<STATE_DIM, MEASUREMENT_DIM, CONTROL_DIM>::step(controlVector control, measurementVector measurement) {
    createSigmaPoints();
    update(control);
    predict(measurement);
    return _state;
}

template<int STATE_DIM, int MEASUREMENT_DIM, int CONTROL_DIM>
Matrix<double, STATE_DIM, 1>
UnscentedKalmanFilter<STATE_DIM, MEASUREMENT_DIM, CONTROL_DIM>::step(controlVector control, measurementVector measurement, double dt) {
    setDt(dt);
    return step(control, measurement);
}

template<int STATE_DIM, int MEASUREMENT_DIM, int CONTROL_DIM>
void UnscentedKalmanFilter<STATE_DIM, MEASUREMENT_DIM, CONTROL_DIM>::createSigmaPoints() {
    _rootFinder.compute(_lambda * _covariance);
    _root = _rootFinder.matrixL();

    _sigmaPoints.col(0) = _state;
    for (int i = 1; i < _state.size()+1; i++) {
        _sigmaPoints.col(i)               = _state + _root.col(i-1);
        _sigmaPoints.col(_state.size()+i) = _state - _root.col(i-1);
    }
}

template<int STATE_DIM, int MEASUREMENT_DIM, int CONTROL_DIM>
void UnscentedKalmanFilter<STATE_DIM, MEASUREMENT_DIM, CONTROL_DIM>::predict(controlVector control) {
    _sigmaPointsF.col(0) = _stateTransfer(_sigmaPoints.col(0), control, _dt);
    _state               = _weights[MEAN_WEIGHT_0] * _sigmaPointsF.col(0);
    for(int i = 1; i < _sigmaPoints.cols(); i++) {
        _sigmaPointsF.col(i)  = _stateTransfer(_sigmaPoints.col(i), control, _dt);
        _state               += _weights[BOTH_WEIGHT_I] * _sigmaPointsF.col(i);
    }

    _covariance = _weights[COVARIANCE_WEIGHT_0] * (_sigmaPointsF.col(0) - _state) * (_sigmaPointsF.col(0) - _state).transpose();
    for(int i = 1; i < _sigmaPoints.cols(); i++) {
        _covariance += _weights[BOTH_WEIGHT_I] * (_sigmaPointsF.col(i) - _state) * (_sigmaPointsF.col(i) - _state).transpose();
    }
    _covariance += _processNoise;
}

template<int STATE_DIM, int MEASUREMENT_DIM, int CONTROL_DIM>
void UnscentedKalmanFilter<STATE_DIM, MEASUREMENT_DIM, CONTROL_DIM>::update(measurementVector measurement) {
    _sigmaPointsH.col(0) = _measurementTransfer(_sigmaPointsF.col(0));
    _measurementState    = _weights[MEAN_WEIGHT_0] * _sigmaPointsH.col(0);
    for(int i = 1; i < _sigmaPoints.cols(); i++) {
        _sigmaPointsH.col(i)  = _measurementTransfer(_sigmaPointsF.col(i));
        _measurementState    += _weights[BOTH_WEIGHT_I] * _sigmaPointsH.col(i);
    }

    _measurementCovariance = _weights[COVARIANCE_WEIGHT_0] * (_sigmaPointsH.col(0) - _measurementState) * (_sigmaPointsH.col(0) - _measurementState).transpose();
    _crossCovariance       = _weights[COVARIANCE_WEIGHT_0] * (_sigmaPointsF.col(0) - _state)            * (_sigmaPointsH.col(0) - _measurementState).transpose();
    for(int i = 1; i < _sigmaPoints.cols(); i++) {
        _measurementCovariance += _weights[BOTH_WEIGHT_I] * (_sigmaPointsH.col(i) - _measurementState) * (_sigmaPointsH.col(i) - _measurementState).transpose();
        _crossCovariance       += _weights[BOTH_WEIGHT_I] * (_sigmaPointsF.col(i) - _state)            * (_sigmaPointsH.col(i) - _measurementState).transpose();
    }
    _measurementCovariance += _measurementNoise;

    _kalmanGain = _crossCovariance * _measurementCovariance.inverse();

    _state      += _kalmanGain * (measurement - _measurementState);
    _covariance -= _kalmanGain * _measurementCovariance * _kalmanGain.transpose();
}

#endif
