//
// Created by liuhongwei on 19-3-26.
//
/*#include <Eigen/StdVector>
#include <Eigen/Core>

#include <iostream>
#include <stdint.h>

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

#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/solvers/pcg/linear_solver_pcg.h"
#include "g2o/types/sba/types_six_dof_expmap.h"

#include "g2o/solvers/structure_only/structure_only_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/optimization_algorithm_dogleg.h"*/

#include <fstream>
#include <string>
#include <sstream>
#include "vertex.h"
using namespace std;
using namespace Eigen;
using namespace g2o;

typedef BlockSolver<BlockSolverTraits<9,3>> block;

int main()
{
    //创建一个线性求解器
    block::LinearSolverType* lsolver=new LinearSolverEigen<block::PoseMatrixType>();
    //创建块求解器
    block* solve_ptr=new block(unique_ptr<block::LinearSolverType>(lsolver));//记得只初始化BlockSolver(std::unique_ptr<LinearSolverType> linearSolver)需要改一下
    //创建总的求解器
    OptimizationAlgorithmLevenberg *solver=new g2o::OptimizationAlgorithmLevenberg(std::unique_ptr<block>(solve_ptr));//同上
    //创建稀疏优化器
    SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);
    //定义图的顶点和边，并添加到优化器中
    //增加相机位姿顶点
    Eigen::Matrix<double,9,1> P;
    P<<0.5,0.5,0.5,1,1,1,300,0.8,1.2;
    for(int i=0;i<49;i++)
    {
        cameravertex* cam=new cameravertex();
        cam->setEstimate(P);
        cam->setId(i);
        optimizer.addVertex(cam);
    }
    //增加路标点的位置顶点
    Eigen::Vector3d d(1,1,1);
    for(int j=0;j<7776;j++)
    {
        pointvertex* point=new pointvertex();
        point->setEstimate(d);
        point->setId(j+49);
        point->setMarginalized(true);
        optimizer.addVertex(point);
    }
    //增加边
    vector<BAedge*> edges;
    //从文件中读入每一行的数据，依次添加到边中
    ifstream is("/home/liuhongwei/workspace/slam/PA10/answer_code/problem-49-7776-pre.txt");//关联文件读入观测数据
    string line;
    int ci,pi;//相机位置索引，路标点索引
    Vector2d e;
    int i=0;
    getline(is,line);
    while(getline(is,line))
    {
        if(i>=31842)
            break;
        BAedge* edge=new BAedge();
        edge->setId(i);
        istringstream it(line);
        it>>ci;//相机索引
        it>>pi;//路标点索引-49
        it>>e[0];
        it>>e[1];
        //设置边对应的两个点
        edge->setVertex(1, dynamic_cast<cameravertex*>(optimizer.vertex(ci)));
        edge->setVertex(0, dynamic_cast<pointvertex*>(optimizer.vertex(pi+49)));
        //设置其协方差矩阵
        edge->setInformation(Matrix2d::Identity());
        //设置边对应的观测值
        edge->setMeasurement(Vector2d(e[0],e[1]));
        //设置核函数
        RobustKernelHuber *r=new RobustKernelHuber();
        r->setDelta(1);
        edge->setRobustKernel( r );
        optimizer.addEdge(edge);
        edges.push_back(edge);
        i++;
    }
    //设置优化参数，开始执行优化
    optimizer.initializeOptimization();
    optimizer.optimize(600);
    //cout<<"T="<<optimizer.vertex(48)->userData()<<endl;
    for(int i=0;i<49;i++)
    {
        cameravertex* c= dynamic_cast<cameravertex *>(optimizer.vertex(i));
        cout<<"vertex id "<<i<<", pos = ";
        VectorXd T=c->estimate();
        cout<<"变换矩阵T="<<endl<<Sophus::SE3::exp(T.block(0,0,6,1)).matrix()<<endl;
        cout<<"f="<<T[6]<<",k1="<<T[7]<<",k2="<<T[8]<<endl;
    }
    for(int j=0;j<7776;j++)
    {
        pointvertex* p= dynamic_cast<pointvertex*>(optimizer.vertex(j+49));
        cout<<"第"<<j<<"个路标点:";
        Vector3d pi=p->estimate();
        cout<<pi[0]<<","<<pi[1]<<","<<pi[2]<<endl;
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
    cout<<"inliers in total points: "<<inliers<<"/"<<31843<<endl;
    optimizer.save("ba.g2o");
    return 0;
}