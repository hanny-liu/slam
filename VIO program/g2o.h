//
// Created by liuhongwei on 19-4-6.
//

#ifndef VIO_PROGRAM_G2O_H
#define VIO_PROGRAM_G2O_H
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/types/slam3d/se3quat.h>
#include <sophus/se3.h>
#include <opencv2/core/core.hpp>
class vertexpose:public g2o::BaseVertex<6,g2o::SE3Quat>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    vertexpose(){}
    //vertexpose():BaseVertex(){}
    ~vertexpose(){}
    //vertexpose(Sophus::SE3 t):T(t){}
    bool read(std::istream &is){ return false;}
    bool write(std::ostream &os) const{ return false;}
    virtual void oplusIml(const number_t *update_)
    {
        Eigen::Map<const Eigen::Matrix<double,6,1>> update(update_);
        setEstimate(g2o::SE3Quat::exp(update)*estimate());
    }
    virtual void setToOriginIml()
    {
        _estimate=g2o::SE3Quat();
    }
};

class edgeerror:public g2o::BaseUnaryEdge<16,Eigen::Matrix<float,16,1>,vertexpose>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    edgeerror(){}
    void edgedirect(Eigen::Vector3d point,float FX,float FY,float CX, float CY,cv::Mat* image)
    {
        x_world=point;
        fx=FX;
        fy=FY;
        cx=CX;
        cy=CY;
        image_=image;
    }
    ~edgeerror(){}

    bool read(std::istream &is){ return false;}
    bool write(std::ostream &os) const{ return false;}
    void computeError()
    {
        const vertexpose* v= static_cast<const vertexpose*>(_vertices[0]);
        Eigen::Vector3d x_local=v->estimate().map(x_world);
        float x=(fx*x_local[0]/x_local[2])+cx;
        float y=(fy*x_local[1]/x_local[2])+cy;

        //判断是否在图像内部
        if(x-2<0||x+1>image_->cols||y-2<0||y+1>image_->rows)
        {
            _error=Eigen::Matrix<float ,16,1>::Zero();
            this->setLevel(1);
        } else
        {
            int m=0;
            for(int i=-2;i<2;i++)
                for(int j=-2;j<2;j++)
                {
                    _error[m]=GetPixelValue(image_,x+i,y+j)-_measurement[m];
                }
        }



    }
    virtual void linearizeOplus()
    {
        if(level()==1)
        {
            _jacobianOplusXi=Eigen::Matrix<float,16,6>::Zero();
            return;
        }
        const vertexpose* p= static_cast<const vertexpose*>(_vertices[0]);
        Eigen::Vector3d x_local=p->estimate().map(x_world);
        float x=x_local[0];
        float y=x_local[1];
        float z=x_local[2];
        Eigen::Matrix<float ,2,3> uq;//像素对投影点偏导
        uq<<fx/z,0,fx*x/z/z,0,y/z,fy*y/z/z;
        Eigen::Matrix<float ,3,6> qt;//投影点对位姿偏导
        qt<<Eigen::Matrix3f::Identity(),Sophus::SO3::exp(Eigen::Vector3d(x,y,z));
        Eigen::RowVector2d iu;//图像梯度
        iu[0]=(GetPixelValue(image_,x+2,y)-GetPixelValue(image_,x-1,y))/3;
        iu[1]=(GetPixelValue(image_,x,y+2)-GetPixelValue(image_,x,y-1))/3;
        for(int i=0;i<16;i++)
        {
            _jacobianOplusXi[i]=iu*uq*qt;
        }
    }

protected:
    inline float GetPixelValue(const cv::Mat &img, float x, float y)
    {
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
private:
    Eigen::Vector3d x_world;
    float fx,fy,cx,cy;
    cv::Mat* image_;
};

#endif //VIO_PROGRAM_G2O_H
