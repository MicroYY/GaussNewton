#include <iostream>

#include <opencv2\opencv.hpp>

#include "GaussNewton.h"


//拟合函数 y = e^(a * x^2 + b * x + c)
// a = 0.5, b = 0.5, c = 2.0

int main(int argc, char** argv)
{
	const double aa = 0.5;
	const double bb = 0.5;
	const double cc = 2.0;
	double a = 0.0;
	double b = 0.0;
	double c = 0.0;

	const size_t N = 100;
	cv::RNG rng(cv::getTickCount());
	double x[N];
	double y[N];
	for (size_t i = 0; i < N; i++)
	{
		x[i] = rng.uniform(0.0, 1.0);
		y[i] = exp(aa * x[i] * x[i] + bb * x[i] + cc) + rng.gaussian(0.05);
	}

	GaussNewton gnSolver(&a, &b, &c);
	gnSolver.solve(x, y, N);
	std::cout << "a = 0.5, b = 0.5, c = 2.0" << std::endl;
	std::cout << "高斯牛顿迭代得到的结果: " << std::endl << "a = " << a << ", b = " << b << ", c = " << c;
}
