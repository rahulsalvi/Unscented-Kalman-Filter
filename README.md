# Unscented Kalman Filter
Implements an Unscented Kalman Filter (UKF) in C++

## Requirements
 - Eigen3
 - Google Test (optional, for unit tests)

## Brief Usage
 - Create an UnscentedKalmanFilter object. Template parameters in order are: State Dimension, Measurement Dimension, Control Dimension
 - Call step() at each time interval, passing in the control and measurement vectors
 - Retrieve state and covariance with state() and covariance()

## Notes
 - Changing values after instantiation should be followed by a call to initialize()

## TODO
 - Documentation
