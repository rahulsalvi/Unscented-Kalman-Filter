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
