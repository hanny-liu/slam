#include <opencv2/opencv.hpp>
#include <sophus/se3.h>
#include <Eigen/Core>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <string>
#include <boost/format.hpp>
#include <pangolin/pangolin.h>


using namespace std;
using namespace Eigen;
using namespace cv;

typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;

// Camera intrinsics
// 内参
double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
// 基线
double baseline = 0.573;
// paths
string left_file = "/home/liuhongwei/workspace/slam/PA8/L6/code/left.png";
string disparity_file = "/home/liuhongwei/workspace/slam/PA8/L6/code/disparity.png";
boost::format fmt_others("/home/liuhongwei/workspace/slam/PA8/L6/code/%06d.png");    // other files

// useful typedefs
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 2, 6> Matrix26d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

// TODO implement this function
/**
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 */
void DirectPoseEstimationMultiLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,
        const vector<double> depth_ref,
        Sophus::SE3 &T21
);

// TODO implement this function
/**
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 */
void DirectPoseEstimationSingleLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,
        const vector<double> depth_ref,
        Sophus::SE3 &T21
);

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

int main(int argc, char **argv) {

    cv::Mat left_img = cv::imread(left_file, 0);//参考图像
    cv::Mat disparity_img = cv::imread(disparity_file, 0);//视差图像

    // let's randomly pick pixels in the first image and generate some 3d points in the first image's frame
    cv::RNG rng;//随机数生成器
    int nPoints = 1000;
    int boarder = 40;
    VecVector2d pixels_ref;
    vector<double> depth_ref;

    // generate pixels in ref and load depth data
    for (int i = 0; i < nPoints; i++) {
        int x = rng.uniform(boarder, left_img.cols - boarder);  // don't pick pixels close to boarder
        int y = rng.uniform(boarder, left_img.rows - boarder);  // don't pick pixels close to boarder
        int disparity = disparity_img.at<uchar>(y, x);
        double depth = fx * baseline / disparity; // you know this is disparity to depth
        depth_ref.push_back(depth);
        pixels_ref.push_back(Eigen::Vector2d(x, y));
    }

    // estimates 01~05.png's pose using this information
    Sophus::SE3 T_cur_ref;//估计位姿

    for (int i = 1; i < 6; i++) {  // 1~10
        cv::Mat img = cv::imread((fmt_others % i).str(), 0);
        //DirectPoseEstimationSingleLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);    // first you need to test single layer
        DirectPoseEstimationMultiLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);
    }
}

void DirectPoseEstimationSingleLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,//像素坐标
        const vector<double> depth_ref,//深度
        Sophus::SE3 &T21//变换矩阵
) {

    // parameters
    int half_patch_size = 4;//窗口大小还是8×8
    int iterations = 100;//迭代次数为100

    double cost = 0, lastCost = 0;
    int nGood = 0;  // good projections
    VecVector2d goodProjection;//好的投影点

    for (int iter = 0; iter < iterations; iter++)
    {
        nGood = 0;
        goodProjection.clear();

        // Define Hessian and bias
        Matrix6d H = Matrix6d::Zero();  // 6x6 Hessian
        Vector6d b = Vector6d::Zero();  // 6x1 bias

        for (size_t i = 0; i < px_ref.size(); i++)
        {

            // compute the projection in the second image
            // TODO START YOUR CODE HERE
            float u =0, v = 0;
            //计算从图一投影到图二的坐标点
            double z1=depth_ref[i];
            double x1=(px_ref[i](0,0)-cx)*z1/fx;
            double y1=(px_ref[i](1,0)-cy)*z1/fy;
            double u1=px_ref[i](0,0);
            double v1=px_ref[i](1,0);
            Vector3d kp(x1,y1,z1);
            kp=T21*kp;
            double x2=kp(0,0);
            double y2=kp(1,0);
            double z2=kp(2,0);
            u=x2*fx/z2+cx;
            v=y2*fy/z2+cy;
            //判断该点是否在图二内部，若是，则添加到goodprojection
            if(u>=0&&u<=img2.cols&&v>=0&&v<=img2.rows)
            {
                nGood++;
                goodProjection.push_back(Eigen::Vector2d(u,v));
            } else
                continue;

            // and compute error and jacobian
            for (int x = -half_patch_size; x < half_patch_size; x++)
                for (int y = -half_patch_size; y < half_patch_size; y++) {

                    double error =0;
                    if(u+x-1>=0&&u+x+1<=img2.cols&&v+y-1>=0&&v+y+1<=img2.rows)
                    {
                        Matrix26d J_pixel_xi;   // pixel to \xi in Lie algebra
                        J_pixel_xi<<fx/z2,0,-fx*x2/z2/z2,-fx*x2*y2/z2/z2,fx+x2*x2*fx/z2/z2,-y2*fx/z2,
                                    0,fy/z2,-fy*y2/z2/z2,-fy-fy*y2*y2/z2/z2,fy*x2*y2/z2/z2,fy*x2/z2;
                        Eigen::RowVector2d J_img_pixel;    // image gradients
                        J_img_pixel(0,0)=(GetPixelValue(img2,u+x+1,v+y)-GetPixelValue(img2,u+x-1,v+y))/2;
                        J_img_pixel(0,1)=(GetPixelValue(img2,u+x,v+y+1)-GetPixelValue(img2,u+x,v+y-1))/2;
                    // total jacobian
                    Vector6d J=-(J_img_pixel*J_pixel_xi).transpose();
                    error=GetPixelValue(img1,u1+x,v1+y)-GetPixelValue(img2,u+x,v+y);
                    H += J * J.transpose();
                    b += -error * J;
                    cost += error * error;
                    } else
                        continue;
                }
            // END YOUR CODE HERE
        }

        // solve update and put it into estimation
        // TODO START YOUR CODE HERE
        Vector6d update;
        update=H.colPivHouseholderQr().solve(b);
        T21 = Sophus::SE3::exp(update) * T21;
        // END YOUR CODE HERE

        cost /= nGood;

        if (isnan(update[0])) {
            // sometimes occurred when we have a black or white patch and H is irreversible
            cout << "update is nan" << endl;
            break;
        }
        if (iter > 0 && cost > lastCost) {
            cout << "cost increased: " << cost << ", " << lastCost << endl;
            break;
        }
        lastCost = cost;
        cout << "cost = " << cost << ", good = " << nGood << endl;
    }
    cout << "good projection: " << nGood << endl;
    cout << "T21 = \n" << T21.matrix() << endl;

    // in order to help you debug, we plot the projected pixels here
    cv::Mat img1_show, img2_show;
    cv::cvtColor(img1, img1_show,COLOR_GRAY2BGR);
    cv::cvtColor(img2, img2_show,COLOR_GRAY2BGR);
    for (auto &px: px_ref) {
        cv::rectangle(img1_show, cv::Point2f(px[0] - 2, px[1] - 2), cv::Point2f(px[0] + 2, px[1] + 2),
                      cv::Scalar(0, 250, 0));
    }
    for (auto &px: goodProjection) {
        cv::rectangle(img2_show, cv::Point2f(px[0] - 2, px[1] - 2), cv::Point2f(px[0] + 2, px[1] + 2),
                      cv::Scalar(0, 250, 0));
    }
    cv::imshow("reference", img1_show);
    cv::imshow("current", img2_show);
    cv::waitKey();
}

void DirectPoseEstimationMultiLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,
        const vector<double> depth_ref,
        Sophus::SE3 &T21
)
{

    // parameters
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    // create pyramids
    vector<cv::Mat> pyr1, pyr2; // image pyramids
    // TODO START YOUR CODE HERE
    for(int i=0;i<pyramids;i++)
    {
        //初始化金字塔图像
        pyr1.emplace_back(Mat((int)(img1.rows*scales[i]),(int)(img1.cols*scales[i]),CV_8UC1));
        pyr2.emplace_back(Mat((int)(img2.rows*scales[i]),(int)(img2.cols*scales[i]),CV_8UC1));
        //给每一层图像赋像素值
        for(int m=0;m<pyr1[i].rows;m++)
            for(int j=0;j<pyr1[i].cols;j++)
            {
                pyr1[i].at<uchar>(m,j)=GetPixelValue(img1,j/scales[i],m/scales[i]);
            }
        for(int m=0;m<pyr2[i].rows;m++)
            for(int n=0;n<pyr2[i].cols;n++)
            {
                pyr2[i].at<uchar>(m,n)=GetPixelValue(img2,n/scales[i],m/scales[i]);
            }
    }
    // END YOUR CODE HERE

    double fxG = fx, fyG = fy, cxG = cx, cyG = cy;  // backup the old values
    for (int level = pyramids - 1; level >= 0; level--)
    {
        VecVector2d px_ref_pyr; // set the keypoints in this pyramid level
        for (auto &px: px_ref)
        {
            px_ref_pyr.push_back(scales[level] * px);
        }

        // TODO START YOUR CODE HERE
        // scale fx, fy, cx, cy in different pyramid levels
        fx=fxG*scales[level];
        fy=fyG*scales[level];
        cx=cxG*scales[level];
        cy=cyG*scales[level];
        // END YOUR CODE HERE
        DirectPoseEstimationSingleLayer(pyr1[level], pyr2[level], px_ref_pyr, depth_ref, T21);
    }

}
