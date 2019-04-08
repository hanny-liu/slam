//
// Created by xiang on 12/21/17.
//

#include <Eigen/Core>
#include <Eigen/Dense>

using namespace Eigen;

#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>

#include "sophus/se3.h"

using namespace std;

typedef vector<Vector3d, Eigen::aligned_allocator<Vector3d>> VecVector3d;
typedef vector<Vector2d, Eigen::aligned_allocator<Vector3d>> VecVector2d;
typedef Matrix<double, 6, 1> Vector6d;



int main(int argc, char **argv) {

    string p3d_file = "/home/liuhongwei/workspace/slam/PA7/【作业】特征点法视觉里程计/code/p3d.txt";
    string p2d_file = "/home/liuhongwei/workspace/slam/PA7/【作业】特征点法视觉里程计/code/p2d.txt";

    VecVector2d p2d;
    VecVector3d p3d;
    Matrix3d K;
    double fx = 520.9, fy = 521.0, cx = 325.1, cy = 249.7;
    K << fx, 0, cx,
         0, fy, cy,
         0, 0, 1;

    // load points in to p3d and p2d 
    // START YOUR CODE HERE
    ifstream v2(p2d_file);
    ifstream v3(p3d_file);
    string line;
    //加载2d点
    while(getline(v2,line))
    {
        istringstream is(line);
        Vector2d point;
        is>>point[0];
        is>>point[1];
        p2d.push_back(point);
    }
    while(getline(v3,line))
    {
        istringstream is(line);
        Vector3d point;
        is>>point[0];
        is>>point[1];
        is>>point[2];
        //cout<<point[0]<<","<<point[1]<<","<<point[2]<<endl;
        p3d.push_back(point);
    }
    // END YOUR CODE HERE
    assert(p3d.size() == p2d.size());//检查P3D和P2D的元素个数是否相等

    int iterations = 100;//迭代100次
    double cost = 0, lastCost = 0;//前一次误差，后一次误差
    int nPoints = p3d.size();//3d点的数量
    cout << "points: " << nPoints << endl;

    Sophus::SE3 T_esti; // estimated pose估计的位姿变量李群

    for (int iter = 0; iter < iterations; iter++)
    {

        Matrix<double, 6, 6> H = Matrix<double, 6, 6>::Zero();//JT×J
        Vector6d b = Vector6d::Zero();

        cost = 0;//  计算误差
        // compute cost
        for (int i = 0; i < nPoints; i++)
        {
            // compute cost for p3d[I] and p2d[I]计算3d到2d误差
            // START YOUR CODE HERE
            Matrix3d se3=T_esti.matrix().block(0,0,3,3);
            Vector3d U=K*se3*p3d[i]/p3d[i][2];
            Vector3d u;
            u << p2d[i][0],p2d[i][1],1;
            Vector2d e=(u-U).head(2);
            cost+=(e[0]*e[0]+e[1]*e[1]);
            //cost+=e[0]+e[1];
	    // END YOUR CODE HERE

	    // compute jacobian计算第iter次迭代的雅克比矩阵
            Matrix<double, 2, 6> J;
            // START YOUR CODE HERE
            //计算从空间坐标系转换到相机坐标系的3d点的相机坐标x',y',z'
            Vector3d p=se3*p3d[i];
            J<<-fx/p[2],0,fx*p[0]/p[2]/p[2],fx*p[0]*p[1]/p[2]/p[2],-fx-fx*p3d[i][0]*p3d[i][0]/p[2]/p[2],fx*p[1]/p[2],
               0,-fy/p[2],fy*p[1]/p[2]/p[2],fy+fy*p[1]/p[2],-fy*p[1]*p[1]/p[2]/p[2],-fy*p[0]/p[2]/p[2];
	    // END YOUR CODE HERE

            H += J.transpose() * J;//将每一个点的J矩阵都相加
            b += -J.transpose() * e;
        }

	// solve dx 
        Vector6d dx;//李代数的增量，需要^一下

        // START YOUR CODE HERE
        dx=H.colPivHouseholderQr().solve(b);//QR分解接触dx
        // END YOUR CODE HERE

        if (isnan(dx[0])) {
            cout << "result is nan!" << endl;
            break;
        }

        if (iter > 0 && cost >= lastCost) {
            // cost increase, update is not good
            cout << "cost: " << cost << ", last cost: " << lastCost << endl;
            break;
        }

        // update your estimation
        // START YOUR CODE HERE
        T_esti= Sophus::SE3::exp(dx)*T_esti;
        // END YOUR CODE HERE
        
        lastCost = cost;

        cout << "iteration " << iter << " cost=" << cout.precision(12) << cost << endl;
    }

    cout << "estimated pose: \n" << T_esti.matrix() << endl;
    return 0;
}
