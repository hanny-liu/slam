#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.h>
using namespace std;
using namespace Eigen;
int main() {
    //装ei向量模的vector
    vector<double> poses1;
    //vector<Sophus::se3, Eigen::aligned_allocator<Sophus::se3>> poses2;
    //读取真实轨迹
    ifstream isl("/home/liuhongwei/workspace/slam/PA3/【作业】李群与李代数/code/groundtruth.txt");
    //读取估计轨迹
    ifstream isr("/home/liuhongwei/workspace/slam/PA3/【作业】李群与李代数/code/estimated.txt");
    string line1,line2;//读取一行
    while(getline(isl,line1)&&getline(isr,line2))//读取真实轨迹和估计轨迹一行的数据
    {
        string n1,n2,n3,n4,t1,t2,t3;//设定读入变量值
        istringstream is1(line1);
        istringstream is2(line2);
        //line1分别读入到变量中
        is1>>n1;
        is1>>t1;is1>>t2;is1>>t3;
        is1>>n1;is1>>n2;is1>>n3;is1>>n4;
        //对真实轨迹的平移部分和旋转部分进行赋值
        Eigen::Vector3d tr1(stod(t1),stod(t2),stod(t3));
        Eigen::Quaterniond q1(stod(n4),stod(n1),stod(n2),stod(n3));
        is2>>n1;
        is2>>t1;is2>>t2;is2>>t3;
        is2>>n1;is2>>n2;is2>>n3;is2>>n4;
        Vector3d tr2(stod(t1),stod(t2),stod(t3));
        Eigen::Quaterniond q2(stod(n4),stod(n1),stod(n2),stod(n3));
        //将四元数转换为变换矩阵SE3
        Sophus::SE3 T1(q1,tr1);
        Sophus::SE3 T2(q2,tr2);
        //求误差矩阵，利用李群李代数的映射关系求出李代数的向量值ei
        Matrix<double,6,1> EI=(T1.inverse()*T2).log();
        double ei=EI.transpose()*EI;
        //将ei存入vector容器中
        poses1.push_back(ei);
    }
    //对容器中每个元素进行平方，然后再求和取平均
    double aver=sqrt(accumulate(poses1.begin(),poses1.end(),0.0)/(poses1.size()));
    cout<<aver<<endl;
    return 0;
}