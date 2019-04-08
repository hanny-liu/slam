//
// Created by liuhongwei on 19-3-26.
//
#include <Eigen/Core>
#include <Eigen/StdVector>


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

//#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/solvers/pcg/linear_solver_pcg.h"
#include "g2o/types/sba/types_six_dof_expmap.h"

#include "g2o/solvers/structure_only/structure_only_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/optimization_algorithm_dogleg.h"
#include <sophus/so3.h>
#include <sophus/se3.h>

#ifndef ANSWER_CODE_VERTEX_H
#define ANSWER_CODE_VERTEX_H

//定义相机位姿顶点类,包含相机内参的优化变量
class cameravertex:public g2o::BaseVertex<9,Eigen::Matrix<double,9,1>>//角轴和平移向量x,y,z,以及相机内参f以及畸变系数k1,k2
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    cameravertex(){}//默认构造函数
    virtual bool read(std::istream& is){ return false;}
    virtual bool write(std::ostream &os)const{return false;}
    virtual void setToOriginImpl(){}//重置,到时候用setEstimate设置初始值
    virtual void oplusImpl(const double *update) override//更新顶点
    {
        Eigen::VectorXd::ConstMapType v(update,cameravertex::Dimension);
        _estimate.block(0,0,3,1)=v.block(0,0,3,1)*_estimate.block(0,0,3,1);
        _estimate.block(3,0,6,1)+=v.block(3,0,6,1);
    }
};
//定义路标顶点类
class pointvertex:public g2o::BaseVertex<3,Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    pointvertex(){}
    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream &os) const{return false;}
    virtual void setToOriginImpl(){}
    virtual void oplusImpl(const double *update) override
    {
        Eigen::Vector3d::ConstMapType v(update);
        _estimate+=v;
    }
};
//定义误差项顶点类

class BAedge:public g2o::BaseBinaryEdge<2,Eigen::Vector2d,pointvertex,cameravertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    BAedge(){}
    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream &os)const{return false;}
    virtual void computeError() override
    {
        const cameravertex *v= dynamic_cast<const cameravertex*>(_vertices[1]);
        const pointvertex *p= dynamic_cast<const pointvertex*>(_vertices[0]);
        Eigen::Vector2d obs(_measurement);
        const Eigen::Vector3d T=v->estimate().block(0,0,3,1);//R
        const Eigen::Vector3d tr=v->estimate().block(3,0,3,1);//t
        Eigen::Vector3d d=Sophus::SO3::exp(T)*p->estimate()+tr;//p
        Eigen::Vector2d c=(-d/d[2]).block(0,0,2,1);//p'
        double k1=v->estimate()(7,0);//k1
        double k2=v->estimate()(8,0);//k2
        double f=v->estimate()(6,0);//f
        double rp=1+k1*(c[0]*c[0]+c[1]*c[1])+k2*(c[0]*c[0]+c[1]*c[1])*(c[0]*c[0]+c[1]*c[1]);//rp
        Eigen::Vector2d y=f*rp*c;//estimate
        _error=obs-y;
    }
    virtual void linearizeOplus() override
    {
        const cameravertex *cam= dynamic_cast<const cameravertex *>(_vertices[1]);
        const pointvertex *point= dynamic_cast<const pointvertex *>(_vertices[0]);
        const Eigen::Vector3d xyz=point->estimate();//路标点
        /*double x=xyz[0];
        double y=xyz[1];
        double z=xyz[2];*/
        Eigen::Vector3d R=cam->estimate().block(0,0,3,1);//旋转矩阵
        Eigen::Vector3d t=cam->estimate().block(3,0,3,1);//平移向量
        Eigen::Matrix4d T;
        T<<Sophus::SO3::exp(R).matrix(),t,
           0,0,0,1;
        Eigen::Vector3d p=(Sophus::SO3::exp(R)).matrix()*xyz+t;//P
        double x=p[0];
        double y=p[1];
        double z=p[2];
        Eigen::Vector2d p1=(-p/p[2]).block(0,0,2,1);//P'
        double x1=p1[0];
        double y1=p1[1];
        double k1=cam->estimate()(7,0);//k1
        double k2=cam->estimate()(8,0);//k2
        double rp=1+k1*(x1*x1+y1*y1)+k2*(x1*x1+y1*y1)*(x1*x1+y1*y1);//rp
        double f=cam->estimate()(6,0);//f
        Eigen::Vector2d p2=f*rp*p1;
        //偏e/偏p'
        Eigen::Matrix2d ep;
        ep<<f*rp,0,
            0,f*rp;
        //偏e/偏rp
        Eigen::Vector2d erp;
        erp<<f*x1,f*y1;
        //偏p'/偏p
        Eigen::Matrix<double,2,3> pp;
        pp<<-1/z,0,x/z/z,
             0,-1/z,y/z/z;
        //偏p/偏路标点
        Eigen::Matrix3d px;
        px=T.block(0,0,3,3);
        //偏p/偏T
        Eigen::Matrix<double,3,6> pr;
        pr<<Eigen::Matrix3d::Identity(),-Sophus::SO3::hat(Sophus::SO3::exp(R)*xyz+t).matrix();
        //偏rp/偏p'
        Eigen::RowVector2d rpp(2*k1*x1+4*k2*x1*x1*x1+4*k2*y1*y1*x1,2*k1*y1+4*k2*y1*y1*y1+4*k2*x1*x1*y1);
        //_jacobianOplusXi=-ep*pp*px;
        _jacobianOplusXi=-ep*pp*px-erp*rpp*pp*px;
        //位姿偏导
        Eigen::Matrix<double,2,6> po;
        //po=-(ep*pp*pr);
        po=-(ep*pp*pr+erp*rpp*pp*pr);
        _jacobianOplusXj(0,0)=po(0,0);
        _jacobianOplusXj(0,1)=po(0,1);
        _jacobianOplusXj(0,2)=po(0,2);
        _jacobianOplusXj(0,3)=po(0,3);
        _jacobianOplusXj(0,4)=po(0,4);
        _jacobianOplusXj(0,5)=po(0,5);
        _jacobianOplusXj(1,0)=po(1,0);
        _jacobianOplusXj(1,1)=po(1,1);
        _jacobianOplusXj(1,2)=po(1,2);
        _jacobianOplusXj(1,3)=po(1,0);
        _jacobianOplusXj(1,4)=po(1,1);
        _jacobianOplusXj(1,5)=po(1,2);
        _jacobianOplusXj(0,6)=-rp*x1;
        _jacobianOplusXj(1,6)=-rp*y1;
        _jacobianOplusXj(0,7)=-f*x1*(x1*x1+y1*y1);
        _jacobianOplusXj(1,7)=-f*y1*(x1*x1+y1*y1);
        _jacobianOplusXj(0,8)=-f*x1*(x1*x1+y1*y1)*(x1*x1+y1*y1);
        _jacobianOplusXj(1,8)=-f*y1*(x1*x1+y1*y1)*(x1*x1+y1*y1);
    }
};

#endif //ANSWER_CODE_VERTEX_H
