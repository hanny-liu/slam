#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include "g2o/stuff/sampler.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/batch_stats.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/optimization_algorithm_dogleg.h"
#include <chrono>
#include <vector>
#include <sophus/se3.h>
#include <math.h>
#include "g2o.h"
using namespace std;
using namespace Eigen;
using namespace cv;
using namespace Sophus;



// bilinear interpolation
inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
            (1 - xx) * (1 - yy) * data[0] +
            xx * (1 - yy) * data[1] +
            (1 - xx) * yy * data[img.step] +
            xx * yy * data[img.step + 1]
    );
}

int main(int argc, char** argv )
{
   //连接模组，读入数据,需要设置连续读入图片信息，多线程



   //相机内参
   float fx,fy,cx,cy;
   Matrix3d K;
   //1读入相机图像数据,需要结合模组进行如何读入
    Mat img_1 = imread ( argv[1], CV_LOAD_IMAGE_COLOR );
    Mat img_2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );
    Mat img_3 = imread ( argv[3], CV_LOAD_IMAGE_COLOR );
    Mat img_4 = imread ( argv[4], CV_LOAD_IMAGE_COLOR );
    //2.读入imu数据
   //三轴角度，转换为弧度
   Vector3f w(0,0,0);
   //三轴角速度，转换为弧度
   Vector3f _w;
   float t=0.001;
   //三轴加速度
   Vector3f a(0,0,0);
   Vector3f v(0,0,0);

   //前端
   //匹配两张图像的特征点，求出其3d坐标
   vector<KeyPoint> keypoints_1, keypoints_2;//像素坐标系下的关键点坐标
   Mat descriptors_1, descriptors_2;//对应特征点的描述子
   int n=100;
   //FAST(img_1,keypoints_1,20);
   Ptr<FeatureDetector> detector=ORB::create(100,1.2f,8, 31,0,2);//创建OR特征角点
   Ptr<DescriptorExtractor> descriptor=ORB::create(100,1.2f,8, 31,0,2);//提取描述子
   //Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );
   //-- 第一步:检测 Oriented FAST 角点位置
   detector->detect ( img_1,keypoints_1 );
   detector->detect ( img_2,keypoints_2 );
   //-- 第二步:根据角点位置计算 BRIEF 描述子
   descriptor->compute ( img_1, keypoints_1, descriptors_1 );
   descriptor->compute ( img_2, keypoints_2, descriptors_2 );
   //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
   Ptr<FlannBasedMatcher> matcher;
   vector< DMatch > matches;//匹配的特征点向量
   matcher->match( descriptors_1, descriptors_2, matches );
   //-- 第四步:匹配点对筛选
    double min_dist=100, max_dist=0;
   //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = matches[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( matches[i].distance <= max ( 2*min_dist, 30.0 ) )
        {
            matches.push_back ( matches[i] );
        }
    }
    //设置前一帧姿态
    SE3 T0(Matrix3d::Identity(),Vector3d(0,0,0));
    //计算前一帧关键点的3d坐标
    int baseline;//基线
    vector<Vector3d> point;
    for(int i=0;i<matches.size();i++)
    {
         float z=(keypoints_1[matches[i].queryIdx].pt.x-keypoints_2[matches[i].trainIdx].pt.x)*fx/baseline;
         float x=(keypoints_1[matches[i].queryIdx].pt.x-cx)/fx;
         float y=(keypoints_1[matches[i].queryIdx].pt.y-cy)/fy;
         point.emplace_back(x,y,z);
    }
     //对后一帧的姿态进行g2o优化
     //设置g2o求解器
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 1>> DirectBlock;  // 求解的向量是6＊1的
    DirectBlock::LinearSolverType *linearSolver = new g2o::LinearSolverCSparse<DirectBlock::PoseMatrixType>();
    DirectBlock *solver_ptr = new DirectBlock(std::unique_ptr<DirectBlock::LinearSolverType>(linearSolver));
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(std::unique_ptr<DirectBlock>(solver_ptr)); // L-M
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);
     //设置g2o顶点,位姿
     vertexpose * p= new vertexpose();
     p->setEstimate(g2o::SE3Quat(T0.rotation_matrix(),T0.translation()));
     p->setId(0);
     optimizer.addVertex(p);
     //设置g2o边
     for(int i=0;i<point.size();i++)
     {

         edgeerror *edge=new edgeerror();
         //edged(point[i],fx,fy,cx,cy,img_3);
         vertexpose *v= dynamic_cast<vertexpose*>(optimizer.vertex(0));
         cv::Mat *l=&img_3;
         edge->edgedirect(point[i],fx,fy,cx,cy,l);
         edge->setId(i);
         edge->setVertex(0,v);
         Eigen::Matrix<float ,16,1> pv;
         if(keypoints_1[matches[i].queryIdx].pt.x-2>=0&&keypoints_1[matches[i].queryIdx].pt.x+1<=img_1.cols&&keypoints_1[matches[i].queryIdx].pt.y-2>=0&&keypoints_1[matches[i].queryIdx].pt.y+1<=img_1.rows)
         {
             for(int i=0;i<16;i++)
             {
                 pv[i]=GetPixelValue(img_1,keypoints_1[matches[i].queryIdx].pt.x+i%4-2,keypoints_1[matches[i].queryIdx].pt.y+i/4-2)
             }
         } else
             continue;
         edge->setMeasurement(pv);
         edge->setInformation(Matrix<float ,16,16>::Identity());
         optimizer.addEdge(edge);
     }
     //设置优化参数，开始执行优化
     optimizer.initializeOptimization();
     optimizer.optimize(100);
     //判斷当前的图像是否提取为关键帧







     //後端，前端需要給給到後端在i到j時刻的測量值和包含的帧数
     //前端传入的参数
     Matrix3d Ri;//前一关键帧的旋转矩阵
     Matrix3d Rj;//当前关键帧估计的旋转矩阵初值
     Vector3d ti;//前一关键帧的平移向量
     Vector3d tj;//当前关键帧估计的平移向量
     Vector3d vi;//前一关键帧的平移向量
     Vector3d vj;//当前关键帧估计的平移向量
     //定义i-j之间的间隔和加速度、速度、角速度值
     int m;//i-j直接的时间间隔数
     double t0;//采样周期
     Vector3d _v(0,0,0);//ij时间内的速度
     Vector3d _t(0,0,0);//ij时间内的平移
     Sophus::SO3 _R(Eigen::Matrix3d::Identity());//ij时间内的角度，程序以弧度为单位
     vector<Vector3f> aij;//ij时间内的加速度
     vector<Vector3f> wij;//ij时间内的角速度
     //定义在i-j区段下，测量值的速度，平移和角度
     for(int i=0;i<m;i++)
     {
         _v+=aij[i]*t0;
         _t+=(_v*t0+0.5*aij[i]*t0*t0);
         _R=Sophus::SO3::exp(wij[i]*t0)*_R;
     }
     Matrix3d _errorR=_R.Adj()*(Ri.transpose()*Rj.transpose());
     Vector3d _errorv=Ri.transpose()*(vj-vi)-_v;//未考虑重力加速度的影响
     Vector3d _errort=Ri.transpose()*(tj-ti-vi*t0*m)-_t;//未考虑重力加速度的影响






     /*//进行前端优化，求出后一帧位姿
     //SE3 T0(Matrix3d::Identity(),Vector3d(0,0,0));
     //求关键点周围4×4矩阵的灰度值
     for(int i=0;i<matches.size();i++)
     {
         float & y=keypoints_1[matches[i].queryIdx].pt.y;
         float & x=keypoints_1[matches[i].queryIdx].pt.x;
         Vector3d p1=T0*point[i];
         p1/=p1[2];
         float x1=(K*(p1))[0];
         float y1=(K*(p1))[1];
         if(y-2>=0&&y+1<=img_1.rows&&x-2>=0&&x+1<=img_1.cols)
             if(y1-2>=0&&y1+1<=img_1.rows&&x1-2>=0&&x1+1<=img_1.cols)
             {
                 float e=GetPixelValue(img_1,x,y)-GetPixelValue(img_3,x1,y1);
             }


     }
     //定义相机姿态轨迹
   vector<Sophus::SE3,Eigen::aligned_allocator<Sophus::SE3>> pose;
    pose.emplace_back(Matrix4d::Identity());
    Vector3f trans;
    //定义初始值的高斯噪声
    const double mean = 0.0;//均值
    const double stddev = 0.1;//标准差
    std::default_random_engine generator;
    std::normal_distribution<double> dist(mean, stddev);
    for(int j=0;j<3;j++)
    {
        trans[j]=pose[0].translation()[j]+dist(generator);
    }
    pose[0].matrix().block(3,0,3,1)=trans;
   for(int i=1;;i++)//需要设置时间循环条件
   {
       //角度
       w+=_w*t;
       //速度
       v+=a*t;
       //平移向量
       trans=pose[i-1].translation()+v*t+0.5*t*t*a;
       //旋转矩阵,将角度转换为弧度
       Matrix3d R;
       R<<cos(w[0])*cos(w[1]),sin(w[0])*cos(w[2])-cos(w[0])*sin(w[1])*sin(w[2]),sin(w[3])*cos(w[0])+cos(w[0])*sin(w[1])*cos(w[2]),
          -sin(w[0])*cos(w[1]),cos(w[0])*cos(w[2])+sin(w[0])*sin(w[1])*sin(w[2]),cos(w[0])*sin(w[3])-sin(w[0])*sin(w[1])*cos(w[2]),
          -sin(w[1]),-sin(w[2])*cos(w[1]),cos(w[1])*cos(w[2]);
       //运动方程
       pose.emplace_back(SE3(R,trans)*pose[i-1]);



   }*/



    return 0;
}