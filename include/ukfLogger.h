#ifndef UKFLOGGER_H
#define UKFLOGGER_H

#include <string>

#include "ukf.h"

using namespace Eigen;

class UnscentedKalmanFilterLogger {
	public:
		UnscentedKalmanFilterLogger(const UnscentedKalmanFilter &filter) : _filter(filter) {}
		//std::string dump();

		const UnscentedKalmanFilter& filter() const {return _filter;}

		double        lambda()       const {return _filter._lambda;}
		const double* weights()      const {return _filter._weights;}
		MatrixXd      sigmaPoints()  const {return _filter._sigmaPoints;}
		MatrixXd      sigmaPointsH() const {return _filter._sigmaPointsH;}
		LLT<MatrixXd> rootFinder()   const {return _filter._rootFinder;}
		MatrixXd      root()         const {return _filter._root;}

	private:
		const UnscentedKalmanFilter &_filter;
};

#endif //UKFLOGGER_H
