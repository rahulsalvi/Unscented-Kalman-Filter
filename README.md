# Unscented-Kalman-Filter
Implements an Unscented Kalman Filter (UKF) in C++

## Requirements
 - Eigen3
 - Google Test (optional, for unit tests)

## Usage
 - Create an UnscentedKalmanFilter object, passing in the values for the matrices.
 - Call step() at each time interval, passing in the control and measurement vectors
 - Retrieve state and covariance with state() and covariance()

## Notes
 - Changing values after instantiation should be followed by a call to initialize()
 - Compiling with UKF\_DIMENSION\_TESTING will cause the filter to throw exceptions if it detects a dimension mismatch
 - Compiling with LOW\_MEMORY will let the filter use less memory at the expense of less meaningful data for the logger

## TODO
 - Allow different timesteps
 - Finish logger
