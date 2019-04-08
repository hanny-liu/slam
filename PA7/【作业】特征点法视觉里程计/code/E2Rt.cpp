//
// Created by 高翔 on 2017/12/19.
// 本程序演示如何从Essential矩阵计算R,t
//

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>

using namespace Eigen;

#include <sophus/so3.h>
#include <sophus/se3.h>

#include <iostream>

using namespace std;

int main(int argc, char **argv) {

    // 给定Essential矩阵
    Matrix3d E;
    E << -0.0203618550523477, -0.4007110038118445, -0.03324074249824097,
            0.3939270778216369, -0.03506401846698079, 0.5857110303721015,
            -0.006788487241438284, -0.5815434272915686, -0.01438258684486258;

    // 待计算的R,t
    Matrix3d R;
    Vector3d t;

    // SVD and fix sigular values
    // START YOUR CODE HERE
    JacobiSVD<MatrixXd> svd(E,ComputeThinU | ComputeThinV);//SVD分解的类型不能直接定义，因为它的分解不清楚
    auto U=svd.matrixU();
    auto V=svd.matrixV();
    auto X=svd.singularValues();
    X[0]=(X[0]+X[1])/2;
    X[1]=X[0];
    X[2]=0;
    Matrix3d x;
    x<<X[0],0,0,0,X[1],0,0,0,X[2];
    AngleAxisd rz1 ( M_PI/2, Vector3d ( 0,0,1 ) );     //沿 Z 轴旋转 90 度
    AngleAxisd rz2 ( -M_PI/2, Vector3d ( 0,0,1 ) );
    auto RZ1 = rz1.toRotationMatrix();
    auto RZ2=rz2.toRotationMatrix();
    // END YOUR CODE HERE

    // set t1, t2, R1, R2 
    // START YOUR CODE HERE
    Matrix3d t_wedge1=U*RZ1*x*U.transpose();//反对称矩阵
    Matrix3d t_wedge2=U*RZ2.transpose()*x*U.transpose();
    //cout<<t_wedge1<<endl;
    auto R1=U*RZ1.transpose()*V.transpose();
    auto R2=U*RZ2.transpose()*V.transpose();
    // END YOUR CODE HERE

    cout << "R1 = " << R1 << endl;
    cout << "R2 = " << R2 << endl;
    cout << "t1 = " << Sophus::SO3::vee(t_wedge1) << endl;//反对称矩阵转换到向量
    cout << "t2 = " << Sophus::SO3::vee(Sophus::SO3::hat(Sophus::SO3(t_wedge2).log())) << endl;

    // check t^R=E up to scale
    Matrix3d tR = t_wedge1 * R1;
    cout << "t^R = " << tR << endl;

    return 0;
}