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

#ifndef UNSCENTEDKALMANFILTER_H
#define UNSCENTEDKALMANFILTER_H

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Cholesky>
#include <eigen3/Eigen/LU>

#ifdef TESTING
#include <gtest/gtest_prod.h>
#endif

#define MEAN_WEIGHT_0 0
#define COVARIANCE_WEIGHT_0 1
#define BOTH_WEIGHT_I 2

using Eigen::Matrix;
using Eigen::LLT;

template<int STATE_DIM, int MEASUREMENT_DIM, int CONTROL_DIM>
class UnscentedKalmanFilter {
    public:
        typedef Matrix<double, STATE_DIM,       1>               stateVector;
        typedef Matrix<double, MEASUREMENT_DIM, 1>               measurementVector;
        typedef Matrix<double, CONTROL_DIM,     1>               controlVector;
        typedef Matrix<double, STATE_DIM,       STATE_DIM>       stateMatrix;
        typedef Matrix<double, MEASUREMENT_DIM, MEASUREMENT_DIM> measurementMatrix;
        typedef Matrix<double, STATE_DIM,       2*STATE_DIM+1>   sigmaPointMatrix;
        UnscentedKalmanFilter() {};
        UnscentedKalmanFilter(stateVector       initialState,
                              stateMatrix       initialCovariance,
                              stateVector       (*stateTransfer)      (stateVector, controlVector, double),
                              measurementVector (*measurementTransfer)(stateVector),
                              stateMatrix       processNoise,
                              measurementMatrix measurementNoise,
                              double            dt,
                              double            kappa,
                              double            alpha = 0.001,
                              double            beta  = 2);

        stateVector       state()            const {return _state;}
        stateMatrix       covariance()       const {return _covariance;}
        stateMatrix       processNoise()     const {return _processNoise;}
        measurementMatrix measurementNoise() const {return _measurementNoise;}
        double            dt()               const {return _dt;}
        double            kappa()            const {return _kappa;}
        double            alpha()            const {return _alpha;}
        double            beta()             const {return _beta;}

        void setState               (stateVector in)                                              {_state               = in;}
        void setCovariance          (stateMatrix in)                                              {_covariance          = in;}
        void setStateTransfer       (stateVector       (*in)(stateVector, controlVector, double)) {_stateTransfer       = in;}
        void setMeasurementTransfer (measurementVector (*in)(stateVector))                        {_measurementTransfer = in;}
        void setProcessNoise        (stateMatrix in)                                              {_processNoise        = in;}
        void setMeasurementNoise    (measurementMatrix in)                                        {_measurementNoise    = in;}
        void setDt                  (double in)                                                   {_dt                  = in;}
        void setKappa               (double in)                                                   {_kappa               = in;}
        void setAlpha               (double in)                                                   {_alpha               = in;}
        void setBeta                (double in)                                                   {_beta                = in;}

        void initialize();
        stateVector step(controlVector control, measurementVector measurement);
        stateVector step(controlVector control, measurementVector measurement, double dt);
    private:
        stateVector _state;
        stateMatrix _covariance;

        stateVector       (*_stateTransfer)       (stateVector, controlVector, double);
        measurementVector (*_measurementTransfer) (stateVector);

        stateMatrix       _processNoise;
        measurementMatrix _measurementNoise;

        double _dt;

        double _kappa;
        double _alpha;
        double _beta;

        //Private member functions
        void createSigmaPoints();
        void predict(controlVector control);
        void update(measurementVector measurement);

        //Intermediary variables
        double _lambda;
        double _weights[3];	//mean 0, covariance 0, both

        sigmaPointMatrix _sigmaPoints;
        sigmaPointMatrix _sigmaPointsF;
        Matrix<double, MEASUREMENT_DIM, 2*STATE_DIM+1> _sigmaPointsH;

        measurementVector _measurementState;
        measurementMatrix _measurementCovariance;
        Matrix<double, STATE_DIM, MEASUREMENT_DIM> _crossCovariance;

        Matrix<double, STATE_DIM, MEASUREMENT_DIM> _kalmanGain;

        LLT<stateMatrix> _rootFinder;
        stateMatrix     _root;

#ifdef TESTING
        FRIEND_TEST(UnscentedKalmanFilterTester, InitializesCorrectly);
        FRIEND_TEST(UnscentedKalmanFilterTester, CreatesSigmaPointsCorrectly);
        FRIEND_TEST(UnscentedKalmanFilterTester, PredictsCorrectly);
        FRIEND_TEST(UnscentedKalmanFilterTester, UpdatesCorrectly);
#endif
};

#define UKF_FUNCTIONS
#include "UnscentedKalmanFilter.cpp"

#endif //UNSCENTEDKALMANFILTER_H
