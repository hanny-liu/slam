#include <iostream>
#include<Eigen/Core>
#include<Eigen/Dense>
#include <Eigen/Cholesky>
#include <Eigen/QR>

using namespace std;
using namespace Eigen;

int main()
{
    MatrixXd mat=MatrixXd::Random(100,100);
    MatrixXd v=MatrixXd::Random(100,1);
    //QR分解
    Matrix<double,100,1> x=mat.colPivHouseholderQr().solve(v);
    cout<<"this is solution \n x="<<"\n"<<x<<endl;
    //Cholesky分解,由于cholesky分解的系数矩阵要求是对称的正定矩阵，因此首先将矩阵对称化同时对系数取绝对值
    MatrixXd a(MatrixXd::Random(100,100));
    MatrixXd v1(MatrixXd::Random(100,1));
    a.topRightCorner(99,99)=a.bottomLeftCorner(99,99);
    a=a.cwiseAbs();
    cout<<a<<endl;
    Matrix<double ,100,100> L = a.llt().matrixL(); //L为对角元素均大于零的下三角矩阵
    Matrix<double ,100,1> y=a.llt().solve(v1);
    cout<<"this is solution \n y="<<"\n"<<y<<endl;

    return 0;
}