//
// Created by liuhongwei on 19-3-16.
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
#include <pangolin/pangolin.h>

using namespace std;


typedef vector<Vector3d, Eigen::aligned_allocator<Vector3d>> VecVector3d;
typedef Matrix<double, 6, 1> Vector6d;
void DrawTrajectory(vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>>,vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>>);

int main()
{
    //读入文件
    string file="/home/liuhongwei/workspace/slam/PA7/【作业】特征点法视觉里程计/code/compare.txt";
    ifstream in(file);
    string line;
    VecVector3d p;//轨迹1的点集
    VecVector3d q;//轨迹2的点集
    Vector3d point;//点
    //读入文件中的3D点
    while(getline(in,line))
    {
        istringstream is(line);
        string word;
        int i=1;
        while(i<12)
        {
            if(i>1&&i<5)
            {
                is>>point[0];
                is>>point[1];
                is>>point[2];
                i+=3;
                //cout<<point;
                p.push_back(point);
            }
            else if(i>9&&i<13)
            {
                is>>point[0];
                is>>point[1];
                is>>point[2];
                i+=3;
                //cout<<point;
                q.push_back(point);
            }
            else
            {
                is>>word;
                i++;
            }

        }

    }
    //求质心
   /* double p1(0),p2(0),p3(0),q1(0),q2(0),q3(0);
    for(auto i:p)
    {
        p1+=i[0];
        p2+=i[1];
        p3+=i[2];
    }
    for(auto j:q)
    {
        q1+=j[0];
        q2+=j[1];
        q3+=j[2];
    }*/
    //计算质心
   /* Vector3d pp,qq;
    pp[0]=p1/p.size();
    pp[1]=p2/p.size();
    pp[2]=p3/p.size();
    qq[0]=q1/q.size();
    qq[1]=q2/q.size();
    qq[2]=q3/q.size();
    //计算去质心坐标
    for(auto & i:p)
    {
        i-=pp;
    }
    for(auto &j:q)
    {
        j-=qq;
    }*/
    //定义变换矩阵T
    Sophus::SE3 T(Matrix3d::Zero(),Vector3d(0,0,0));
    int iterations =100;//迭代100次
    double cost = 0, lastCost = 0;//前一次误差，后一次误差
    //Sophus库的扰动模型，求导
    for (int iter = 0; iter < iterations; iter++)
    {

        Matrix<double, 6, 6> H = Matrix<double, 6, 6>::Zero();//JT×J
        Matrix<double,6,1> b = Matrix<double,6,1>::Zero();

        cost = 0;//  计算误差
        // compute cost
        for (int i = 0; i < p.size(); i++)
        {
            // compute cost for p3d[I] and p2d[I]计算3d到2d误差
            // START YOUR CODE HERE
            Vector3d e=q[i]-T*p[i];
            cost+=(e[0]*e[0]+e[1]*e[1]+e[2]*e[2]);
            // END YOUR CODE HERE

            // compute jacobian计算第iter次迭代的雅克比矩阵
            Matrix<double, 3, 6> J;
            // START YOUR CODE HERE
            //计算从空间坐标系转换到相机坐标系的3d点的相机坐标x',y',z'
            J<<1,0,0,0,q[i][2],-q[i][1],
               0,1,0,-q[i][2],0,q[i][0],
               0,0,1,q[i][1],-q[i][0],0;
            // END YOUR CODE HERE

            H += J.transpose() * J;//将每一个点的J矩阵都相加
            b += -J.transpose() * e;
        }
        // solve dx
        Vector6d dx;//李代数的增量，需要^一下

        // START YOUR CODE HERE
        dx=H.householderQr().solve(b);//QR分解接触dx
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
        T= Sophus::SE3::exp(dx)*T;
        // END YOUR CODE HERE

        lastCost = cost;

        cout << "iteration " << iter << " cost=" << cout.precision(12) << cost << endl;
    }
    cout << "estimated pose: \n" << T.matrix() << endl;
    //在pangolin上画图，轨迹点变化但轨迹的四元数不会变化
    vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses1;
    vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses2;
    /*for(auto & i:p)
    {
        i=i+pp;
    }
    for(auto & j:q)
    {
        j=j+qq;
    }*/
    ofstream os1("/home/liuhongwei/workspace/slam/PA7/【作业】特征点法视觉里程计/code/data1.txt");
    ofstream os2("/home/liuhongwei/workspace/slam/PA7/【作业】特征点法视觉里程计/code/data2.txt");

    for(auto i:p)
    {
        os1<<T*i;
    }
    for(auto j:q)
    {
        os2<<j;
    }

    Vector4d r1,r2;
    //读入两个轨迹的四元数
    string file1="/home/liuhongwei/workspace/slam/PA7/【作业】特征点法视觉里程计/code/compare.txt";
    ifstream on(file1);
    int j=0;
    while(getline(on,line))
    {
        istringstream is(line);
        string word;
        int i=1;
        while(i<17)
        {
            if(i>4&&i<9)
            {
                is>>r1[0];
                is>>r1[1];
                is>>r1[2];
                is>>r1[3];
                i+=4;
                Quaterniond R1(r1[1],r1[2],r1[3],r1[0]);
                AngleAxisd A1(R1);
                poses1.emplace_back(A1.toRotationMatrix(),p[j]);
            }
            else if(i>=13&&i<=16)
            {
                is>>r2[0];
                is>>r2[1];
                is>>r2[2];
                is>>r2[3];
                i+=4;
                Quaterniond R2(r2[1],r2[2],r2[3],r2[0]);
                AngleAxisd A2(R2);
                poses2.emplace_back(A2.toRotationMatrix(),q[j]);
            }
            else
            {
                is>>word;
                i++;
            }

        }
        j++;

    }
    for(int i=0;i<poses1.size();i++)
    {
        poses1[i]=T*poses1[i];
    }

    double p1(0),p2(0),p3(0),q1(0),q2(0),q3(0);
    for(auto i:poses1)
    {
        p1+=i.translation()[0];
        p2+=i.translation()[1];
        p3+=i.translation()[2];
    }
    for(auto j:poses2)
    {
        q1+=j.translation()[0];
        q2+=j.translation()[1];
        q3+=j.translation()[2];
    }
    //计算质心
     Vector3d pp,qq;
     pp[0]=p1/p.size();
     pp[1]=p2/p.size();
     pp[2]=p3/p.size();
     qq[0]=q1/q.size();
     qq[1]=q2/q.size();
     qq[2]=q3/q.size();
     //计算尺度
     VecVector3d p11,q11;
     for(int i=0;i<p.size();i++)
     {
         p11.push_back((p[i]-pp));
     }
     for(int n=0;n<p.size();n++)
     {
         q11.push_back((q[n]-qq));
     }
     float k=0;
     for(int i=0;i<p11.size();i++)
     {
         float m=sqrt(p11[i][0] * p11[i][0] + p11[i][1] * p11[i][1] + p11[i][2] * p11[i][2]);
         float n=sqrt(q11[i][0] *q11[i][0] + q11[i][1] * q11[i][1] + q11[i][2] * q11[i][2]);
         k+=m/n;
     }
     k/=p11.size();
    for(auto i:poses1)
    {
        i=Sophus::SE3(i.rotation_matrix()/k,(qq-i*pp/k));
    }
     /*for(auto & i:poses1)
     {
         i=Sophus::SE3(i.rotation_matrix()/k,(-T+(qq-pp)));
     }*/
    DrawTrajectory(poses1,poses2);
    //DrawTrajectory(poses2);
    return 0;
}

/*******************************************************************************************/

void DrawTrajectory(vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses1,vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses2) {
    if (poses1.empty()&&poses2.empty()) {
        cerr << "Trajectory is empty!" << endl;
        return;
    }

    // create pangolin window and plot the trajectory
    //创建一个窗口
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    //启动深度测试
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    // Define Projection and initial ModelView matrix
    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );
    //看SimpleDisplay中边界的设置就知道
    //setBounds 跟opengl的viewport 有关
    // Create Interactive View in window:handler3D
    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));


    while (pangolin::ShouldQuit() == false) {
        // Clear screen and activate view to render into
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);


        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glLineWidth(2);
        for (size_t i = 0; i < poses1.size() - 1; i++) {
            glColor3f(1 - (float) i / poses1.size(), 0.0f, (float) i / poses1.size());
            glBegin(GL_LINES);
            auto p1 = poses1[i], p2 = poses1[i + 1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }
        for (size_t i = 0; i < poses2.size() - 1; i++) {
            glColor3f(1 - (float) i / poses2.size(), 0.0f, (float) i / poses2.size());
            glBegin(GL_LINES);
            auto p1 = poses2[i], p2 = poses2[i + 1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }
        // Swap frames and Process Events
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }

}