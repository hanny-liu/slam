#include <iostream>
#include <cmath>
using namespace std;

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

using namespace Eigen;

int main() {

    Quaterniond q1 (0.55,0.3,0.2,0.2) ;
    Quaterniond q2 (-0.1,0.3, -0.7, 0.2 );
    //对q1,q2进行初始化
    q1.normalize();
    q2.normalize();
    //定义p坐标
    Vector3d px(0.5,-0.1,0.2);
    //将p从q1转换到世界坐标系
    Vector3d px1=q1.inverse()*(px-Vector3d(0.7, 1.1, 0.2));
    cout<<px1<<endl;
    //将p从世界坐标系转换到q2坐标系
    Vector3d p12=q2*(px1)+Vector3d(-0.1, 0.4, 0.8);
    cout<<p12<<endl;
    return 0;
}