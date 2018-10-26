#pragma once
#include <Eigen\Eigen>


class GaussNewton
{
public:
	GaussNewton(double* _aa, double* _bb, double* _cc);

	void solve(double* _x, double* _y,size_t _num, int _maxIter = 1000, double _epsilon = 1e-10);
	
private:
	double *aa, *bb, *cc;
	size_t num;
	double *x, *y;

	void compute_Jacobi_Fx(Eigen::MatrixXd& _jacobi, Eigen::MatrixXd& _fx);
};