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

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Cholesky>
#include <eigen3/Eigen/LU>
#include <vector>
#include <exception>

#ifdef TESTING
#include <gtest/gtest_prod.h>
#endif

using namespace Eigen;

#define MEAN_WEIGHT_0  0
#define COVARIANCE_WEIGHT_0  1
#define BOTH_WEIGHT_I 2

class UnscentedKalmanFilter {
    public:
        UnscentedKalmanFilter() {};
        UnscentedKalmanFilter(VectorXd initialState,
                              MatrixXd initialCovariance,
                              VectorXd (*stateTransfer)(VectorXd, VectorXd, double),
                              VectorXd (*measurementTransfer)(VectorXd),
                              MatrixXd processNoise,
                              MatrixXd measurementNoise,
                              double   dt,
                              double   kappa,
                              double   alpha = 0.001,
                              double   beta  = 2);

        VectorXd state()            const {return _state;}
        MatrixXd covariance()       const {return _covariance;}
        MatrixXd processNoise()     const {return _processNoise;}
        MatrixXd measurementNoise() const {return _measurementNoise;}
        double   dt()               const {return _dt;}
        double   kappa()            const {return _kappa;}
        double   alpha()            const {return _alpha;}
        double   beta()             const {return _beta;}

        void setState               (VectorXd in)                                {_state               = in;}
        void setCovariance          (MatrixXd in)                                {_covariance          = in;}
        void setStateTransfer       (VectorXd (*in)(VectorXd, VectorXd, double)) {_stateTransfer       = in;}
        void setMeasurementTransfer (VectorXd (*in)(VectorXd))                   {_measurementTransfer = in;}
        void setProcessNoise        (MatrixXd in)                                {_processNoise        = in;}
        void setMeasurementNoise    (MatrixXd in)                                {_measurementNoise    = in;}
        void setDt                  (double in)                                  {_dt                  = in;}
        void setKappa               (double in)                                  {_kappa               = in;}
        void setAlpha               (double in)                                  {_alpha               = in;}
        void setBeta                (double in)                                  {_beta                = in;}

        void initialize();
        void step(VectorXd control, VectorXd measurement);
    private:
        VectorXd _state;
        MatrixXd _covariance;

        VectorXd (*_stateTransfer)(VectorXd, VectorXd, double);
        VectorXd (*_measurementTransfer)(VectorXd);

        MatrixXd _processNoise;
        MatrixXd _measurementNoise;

        double   _dt;

        double   _kappa;
        double   _alpha;
        double   _beta;

        //Private member functions
        void     fixMatrixSizes();
        void     calculateConstants();

        void     createSigmaPoints();
        void     predict(VectorXd control);
        void     update(VectorXd measurement);

        //Intermediary variables
        friend class UnscentedKalmanFilterLogger; //in case we need access to these externally

        double   _lambda;
        double _weights[3];	//mean 0, covariance 0, both

        MatrixXd _sigmaPoints;
#ifndef LOW_MEMORY
        MatrixXd _sigmaPointsF;
#endif
        MatrixXd _sigmaPointsH;

        VectorXd _measurementState;
        MatrixXd _measurementCovariance;
        MatrixXd _crossCovariance;

        MatrixXd _kalmanGain;

        LLT<MatrixXd> _rootFinder;
        MatrixXd _root;

#ifdef TESTING
        FRIEND_TEST(UnscentedKalmanFilterTester, InitializesCorrectly);
        FRIEND_TEST(UnscentedKalmanFilterTester, CreatesSigmaPointsCorrectly);
        FRIEND_TEST(UnscentedKalmanFilterTester, PredictsCorrectly);
        FRIEND_TEST(UnscentedKalmanFilterTester, UpdatesCorrectly);
#endif
};

#endif //UKF_H
