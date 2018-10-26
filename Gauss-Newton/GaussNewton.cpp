#include <iostream>

#include "GaussNewton.h"

GaussNewton::GaussNewton(double * _aa, double * _bb, double * _cc)
	:aa(_aa), bb(_bb), cc(_cc)
{

}

void GaussNewton::solve(double * _x, double * _y, size_t _num, int _maxIter, double _epsilon)
{
	num = _num;
	x = _x;
	y = _y;
	bool is_convergent = false;
	for (size_t i = 0; i < _maxIter; i++)
	{
		Eigen::MatrixXd jacobi;
		Eigen::MatrixXd fx;
		compute_Jacobi_Fx(jacobi,fx);	
		Eigen::MatrixXd H = jacobi.transpose() * jacobi;
		Eigen::MatrixXd B = -jacobi.transpose() * fx;
		Eigen::VectorXd delta_x = H.ldlt().solve(B);
		if (delta_x.norm() < _epsilon)
		{
			is_convergent = true;
			break;
		}
		*aa += delta_x(0);
		*bb += delta_x(1);
		*cc += delta_x(2);
	}
	if (is_convergent)
		std::cout << "Converged!" << std::endl;
	else
		std::cout << "Diverged!" << std::endl;

}

void GaussNewton::compute_Jacobi_Fx(Eigen::MatrixXd& _jacobi,Eigen::MatrixXd& _fx)
{
	_jacobi.resize(num, 3);
	_fx.resize(num, 1);
	for (size_t i = 0; i < num; i++)
	{
		_jacobi(i, 0) = -x[i] * x[i] * exp(*aa * x[i] * x[i] + *bb * x[i] + *cc);
		_jacobi(i, 1) = -x[i] *        exp(*aa * x[i] * x[i] + *bb * x[i] + *cc);
		_jacobi(i, 2) =               -exp(*aa * x[i] * x[i] + *bb * x[i] + *cc);
		_fx(i, 0)     = y[i] -         exp(*aa * x[i] * x[i] + *bb * x[i] + *cc);
	}
}
