//
// Created by xiang on 1/4/18.
// this program shows how to perform direct bundle adjustment
//
#include <iostream>

using namespace std;

#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/sim3/sim3.h>

#include <Eigen/Core>
#include <sophus/se3.h>
#include <opencv2/opencv.hpp>

#include <pangolin/pangolin.h>
#include <boost/format.hpp>

#include <unordered_set>
#include <memory>
#include <vector>
#include <stdlib.h>

#include "g2o/stuff/sampler.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/batch_stats.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/optimization_algorithm_dogleg.h"

//#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/solvers/pcg/linear_solver_pcg.h"
#include "g2o/types/sba/types_six_dof_expmap.h"

#include "g2o/solvers/structure_only/structure_only_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/optimization_algorithm_dogleg.h"

typedef vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> VecSE3;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVec3d;

// global variables
string pose_file = "/home/liuhongwei/workspace/slam/PA10/L7/code/poses.txt";
string points_file = "/home/liuhongwei/workspace/slam/PA10/L7/code/points.txt";

// intrinsics
float fx = 277.34;
float fy = 291.402;
float cx = 312.234;
float cy = 239.777;

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

// g2o vertex that use sophus::SE3 as pose
class VertexSophus : public g2o::BaseVertex<6, Sophus::SE3> {//相机位姿顶点
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    VertexSophus() {}

    ~VertexSophus() {}

    bool read(std::istream &is) {}

    bool write(std::ostream &os) const {}

    virtual void setToOriginImpl() {
        _estimate = Sophus::SE3();
    }

    virtual void oplusImpl(const double *update_) {
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> update(update_);
        setEstimate(Sophus::SE3::exp(update) * estimate());
    }
};
//定义3d顶点的虚函数
g2o::VertexSBAPointXYZ::VertexSBAPointXYZ(){}
bool g2o::VertexSBAPointXYZ::read(std::istream& is){}
bool g2o::VertexSBAPointXYZ::write(std::ostream& os) const{}

// TODO edge of projection error, implement it
// 16x1 error, which is the errors in patch
typedef Eigen::Matrix<double,16,1> Vector16d;//误差二元边的定义
class EdgeDirectProjection : public g2o::BaseBinaryEdge<16, Vector16d, g2o::VertexSBAPointXYZ, VertexSophus> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeDirectProjection(float *color, cv::Mat &target) {//color表示点对应的4×4块内的每个小块的像素值，在添加边时，一定要添加上去
        this->origColor = color;
        this->targetImg = target;
    }

    ~EdgeDirectProjection() {}

    virtual void computeError() override {
        // TODO START YOUR CODE HERE
        // compute projection error ...
        Eigen::Matrix3d K;
        K<<fx,0,cx,
           0,fy,cy,
           0,0,1;
        const VertexSophus* T= static_cast<const VertexSophus*>(_vertices[1]);//相机位姿T
        const g2o::VertexSBAPointXYZ* P= static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);//3d点位置

        Vector16d e;
        Vector16d colori;
        for(int i=0;i<16;i++)
            e[i]=origColor[i];
        _measurement=e;
        //将特征点从世界坐标系转换到相机坐标系
        Eigen::Vector3d pi=K*(T->estimate()*P->estimate());
        //从相机坐标系转换到归一化坐标系
        Eigen::Vector2d point=(pi/pi[2]).head(2);
        //判断投影点是否在图像内
        if(point[0]-2>=0&&point[0]+1<=targetImg.cols&&point[1]-2>=0&&point[1]+1<=targetImg.rows)
        {
            int n=0;
            for(int i=-2;i<2;i++)
                for(int j=-2;j<2;j++)
                {
                    float x=point[0]+i;
                    float y=point[1]+j;
                    colori[n]=GetPixelValue(targetImg,x,y);
                    n++;
                }
            _error=_measurement-colori;
        }

        // END YOUR CODE HERE
    }

    // Let g2o compute jacobian for you

    virtual bool read(istream &in) {}

    virtual bool write(ostream &out) const {}

private:
    cv::Mat targetImg;  // the target image
    float *origColor = nullptr;   // 16 floats, the color of this point
};

// plot the poses and points for you, need pangolin
void Draw(const VecSE3 &poses, const VecVec3d &points);

int main(int argc, char **argv) {

    // read poses and points
    VecSE3 poses;
    VecVec3d points;
    ifstream fin(pose_file);

    while (!fin.eof()) {
        double timestamp = 0;
        fin >> timestamp;
        if (timestamp == 0) break;
        double data[7];
        for (auto &d: data) fin >> d;
        poses.push_back(Sophus::SE3(
                Eigen::Quaterniond(data[6], data[3], data[4], data[5]),
                Eigen::Vector3d(data[0], data[1], data[2])
        ));
        if (!fin.good()) break;
    }
    fin.close();


    vector<float *> color;
    fin.open(points_file);
    while (!fin.eof()) {
        double xyz[3] = {0};
        for (int i = 0; i < 3; i++) fin >> xyz[i];
        if (xyz[0] == 0) break;
        points.push_back(Eigen::Vector3d(xyz[0], xyz[1], xyz[2]));
        float *c = new float[16];
        for (int i = 0; i < 16; i++) fin >> c[i];
        color.push_back(c);

        if (fin.good() == false) break;
    }
    fin.close();

    cout << "poses: " << poses.size() << ", points: " << points.size() << endl;

    // read images
    vector<cv::Mat> images;
    boost::format fmt("/home/liuhongwei/workspace/slam/PA10/L7/code/%d.png");
    for (int i = 0; i < 7; i++) {
        images.push_back(cv::imread((fmt % i).str(), 0));
    }

    // build optimization problem
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> DirectBlock;  // 求解的向量是6＊1的
    DirectBlock::LinearSolverType *linearSolver = new g2o::LinearSolverDense<DirectBlock::PoseMatrixType>();
    DirectBlock *solver_ptr = new DirectBlock(std::unique_ptr<DirectBlock::LinearSolverType>(linearSolver));
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(std::unique_ptr<DirectBlock>(solver_ptr)); // L-M
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    // TODO add vertices, edges into the graph optimizer
    // START YOUR CODE HERE
    //添加相机位姿顶点集poses
    for(int i=0;i<poses.size();i++)
    {
        VertexSophus * p=new VertexSophus();
        p->setEstimate(poses[i]);
        p->setId(i);
        optimizer.addVertex(p);
    }
    int n=poses.size();
    int v=points.size();
    //添加3D点顶点集points
    for(int j=0;j<points.size();j++)
    {
        g2o::VertexSBAPointXYZ *pl=new g2o::VertexSBAPointXYZ();
        pl->setEstimate(points[j]);
        pl->setId(n+j);
        pl->setMarginalized(true);
        optimizer.addVertex(pl);
    }
    //添加边
    int m=0;
    vector<EdgeDirectProjection*> edges;
    Vector16d e;
    //co<<1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
    //由于计算边的误差函数添加了识别原始数据投影点周围的16个数的像素值，因此需要根据其函数，设置16个点color的像素值
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<points.size();j++)//遍历所有的图片中所有的3d点先判断3d点的投影是否落入每个图像内，若落入则添加边，若未落入则不添加
        {
            Eigen::Vector3d p=poses[i]*points[j];//在i图像中的j点的坐标
            Eigen::Vector2d po=(p/p[2]).head(2);
            Eigen::Vector2d tp(fx*po[0]+cx,fy*po[1]+cy);
            if(tp[0]-2>=0&&tp[0]+1<=images[i].cols&&tp[1]-2>=0&&tp[1]+1<=images[i].rows)
            {
                //cout<<"point"<<j<<":("<<points[j][0]<<","<<points[j][1]<<","<<points[j][2]<<")"<<endl;
                EdgeDirectProjection *edge=new EdgeDirectProjection(color[j],images[i]);
                edge->setId(m);
                edge->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(n+j)));
                edge->setVertex(1, dynamic_cast<VertexSophus*>(optimizer.vertex(i)));
                g2o::RobustKernelHuber *r=new g2o::RobustKernelHuber();
                r->setDelta(1);
                edge->setRobustKernel(r);
                edge->setInformation(Eigen::Matrix<double,16,16>::Identity());
                optimizer.addEdge(edge);
                edges.push_back(edge);
                m++;
            }

        }
    }
    // END YOUR CODE HERE

    // perform optimization

    optimizer.initializeOptimization(0);
    optimizer.optimize(100);

    // TODO fetch data from the optimizer
    // START YOUR CODE HERE
    poses.clear();
    points.clear();
    for(int i=0;i<n;i++)
    {
        VertexSophus* c= dynamic_cast<VertexSophus *>(optimizer.vertex(i));
        cout<<"vertex id "<<i<<", pos = ";
        Eigen::MatrixXd T=c->estimate().matrix();
        cout<<"变换矩阵T="<<endl<<T<<endl;
        poses.emplace_back(T.block(0,0,3,3),T.block(3,0,3,1));
        //cout<<"f="<<T[6]<<",k1="<<T[7]<<",k2="<<T[8]<<endl;
    }
    auto pw=poses;
    pw.erase(pw.begin()+2,pw.end());
    for(int j=0;j<v;j++)
    {
        g2o::VertexSBAPointXYZ* p= dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(j+n));
        cout<<"第"<<j<<"个路标点:";
        Eigen::Vector3d pi=p->estimate();
        points.push_back(pi);
        //cout<<pi[0]<<","<<pi[1]<<","<<pi[2]<<endl;
    }
    int inliers=0;
    for(auto e:edges)
    {
        e->computeError();
        // chi2 就是 error*\Omega*error, 如果这个数很大，说明此边的值与其他边很不相符
        if ( e->chi2() > 1 )
        {
            cout<<"error = "<<e->chi2()<<endl;
        } else
            inliers++;
    }
    cout<<"inliers in total points: "<<inliers<<"/"<<m<<endl;
    optimizer.save("badirectr.g2o");
    // END YOUR CODE HERE

    // plot the optimized points and poses
    //Draw(poses, points);
    Draw(pw, points);

    // delete color data
    for (auto &c: color) delete[] c;
    return 0;
}

void Draw(const VecSE3 &poses, const VecVec3d &points) {
    if (poses.empty() || points.empty()) {
        cerr << "parameter is empty!" << endl;
        return;
    }

    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
            //pangolin::ModelViewLookAt(0, 0, 0, 0, 0, 0, 0.0, 0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));


    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

        // draw poses
        float sz = 0.1;
        int width = 640, height = 480;
        for (auto &Tcw: poses) {
            glPushMatrix();
            Sophus::Matrix4f m = Tcw.inverse().matrix().cast<float>();
            glMultMatrixf((GLfloat *) m.data());
            glColor3f(1, 0, 0);
            glLineWidth(2);
            glBegin(GL_LINES);
            glVertex3f(0, 0, 0);
            glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(0, 0, 0);
            glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(0, 0, 0);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(0, 0, 0);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
            glEnd();
            glPopMatrix();
        }

        // points
        glPointSize(2);
        glBegin(GL_POINTS);
        for (size_t i = 0; i < points.size(); i++) {
            glColor3f(0.0, points[i][2]/4, 1.0-points[i][2]/4);
            glVertex3d(points[i][0], points[i][1], points[i][2]);
        }
        glEnd();

        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
}

