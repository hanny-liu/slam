//
// Created by liuhongwei on 19-3-22.
//

#include <opencv2/opencv.hpp>
#include <string>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace cv;

// this program shows how to use optical flow

string file_1 = "/home/liuhongwei/workspace/slam/PA8/L6/code/left.png";  // first image
string file_2 = "/home/liuhongwei/workspace/slam/PA8/L6/code/right.png";  // second image

// TODO implement this funciton
/**
 * single level optical flow
 * @param [in] img1 the first image
 * @param [in] img2 the second image
 * @param [in] kp1 keypoints in img1
 * @param [in|out] kp2 keypoints in img2, if empty, use initial guess in kp1
 * @param [out] success true if a keypoint is tracked successfully
 * @param [in] inverse use inverse formulation?
 */
void OpticalFlowSingleLevel(
        const Mat &img1,
        const Mat &img2,
        const vector<KeyPoint> &kp1,
        vector<KeyPoint> &kp2,
        vector<bool> &success,
        bool inverse = false
);

// TODO implement this funciton
/**
 * multi level optical flow, scale of pyramid is set to 2 by default
 * the image pyramid will be create inside the function
 * @param [in] img1 the first pyramid
 * @param [in] img2 the second pyramid
 * @param [in] kp1 keypoints in img1
 * @param [out] kp2 keypoints in img2
 * @param [out] success true if a keypoint is tracked successfully
 * @param [in] inverse set true to enable inverse formulation
 */
void OpticalFlowMultiLevel(
        const Mat &img1,
        const Mat &img2,
        const vector<KeyPoint> &kp1,
        vector<KeyPoint> &kp2,
        vector<bool> &success,
        bool inverse = false
);

/**
 * get a gray scale value from reference image (bi-linear interpolated)
 * @param img
 * @param x
 * @param y
 * @return
 */
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


int main()
{
    //读入图像
    Mat img1 = imread(file_1, 0);
    Mat img2 = imread(file_2, 0);

    //提取GIFT特征点

    //利用金字塔LK光流找到匹配的特征点

    // key points, using GFTT here.
    vector<KeyPoint> kp1;
    Ptr<GFTTDetector> detector = GFTTDetector::create(500,0.01,20); // maximum 500 keypoints
    detector->detect(img1, kp1);

    // now lets track these key points in the second image
    // first use single level LK in the validation picture
    /*vector<KeyPoint> kp2_single;
    vector<bool> success_single;
    OpticalFlowSingleLevel(img1, img2, kp1, kp2_single, success_single);*/

    // then test multi-level LK
    vector<KeyPoint> kp2_multi;
    vector<bool> success_multi;
    OpticalFlowMultiLevel(img1, img2, kp1, kp2_multi, success_multi);

    // use opencv's flow for validation
    vector<Point2f> pt1, pt2;
    for (auto &kp: kp1) pt1.push_back(kp.pt);
    vector<uchar> status;
    vector<float> error;
    cv::calcOpticalFlowPyrLK(img1, img2, pt1, pt2, status, error, cv::Size(8, 8));

    // plot the differences of those functions
    /*Mat img2_single;
    cv::cvtColor(img2, img2_single, COLOR_GRAY2BGR);
    for (int i = 0; i < kp2_single.size(); i++) {
        if (success_single[i]) {
            cv::circle(img2_single, kp2_single[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_single, kp1[i].pt, kp2_single[i].pt, cv::Scalar(0, 250, 0));
        }
    }*/


    //根据左右图的特征点像素坐标u坐标计算视差

    vector<double> disparity;
    for(int i=0;i<kp1.size();i++)
    {
        disparity.emplace_back((kp1[i].pt.x-kp2_multi[i].pt.x));
    }




    Mat img2_multi;
    cv::cvtColor(img2, img2_multi, COLOR_GRAY2BGR);
    for (int i = 0; i < kp2_multi.size(); i++) {
        if (success_multi[i]) {
            cv::circle(img2_multi, kp2_multi[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_multi, kp1[i].pt, kp2_multi[i].pt, cv::Scalar(0, 250, 0));
        }
    }

    Mat img2_CV;
    cv::cvtColor(img2, img2_CV, COLOR_GRAY2BGR);
    for (int i = 0; i < pt2.size(); i++) {
        if (status[i]) {
            cv::circle(img2_CV, pt2[i], 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_CV, pt1[i], pt2[i], cv::Scalar(0, 250, 0));
        }
    }
    //cv::imshow("tracked single level", img2_single);
    cv::imshow("tracked multi level", img2_multi);
    cv::imshow("tracked by opencv", img2_CV);
    cv::waitKey(0);

    return 0;
}

void OpticalFlowSingleLevel(
        const Mat &img1,
        const Mat &img2,
        const vector<KeyPoint> &kp1,
        vector<KeyPoint> &kp2,
        vector<bool> &success,
        bool inverse
) {

    // parameters
    int half_patch_size = 4;//窗口半宽
    int iterations = 10;//迭代次数
    bool have_initial = !kp2.empty();//判断图2的关键点是否为空

    for (size_t i = 0; i < kp1.size(); i++)
    {
        auto kp = kp1[i];
        double dx = 0, dy = 0; // dx,dy need to be estimated
        if (have_initial)
        {
            dx = kp2[i].pt.x - kp.pt.x;
            dy = kp2[i].pt.y - kp.pt.y;
        }

        double cost = 0, lastCost = 0;
        bool succ = true; // indicate if this point succeeded

        // Gauss-Newton iterations
        for (int iter = 0; iter < iterations; iter++)
        {
            Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
            Eigen::Vector2d b = Eigen::Vector2d::Zero();
            cost = 0;

            if (kp.pt.x + dx <= half_patch_size || kp.pt.x + dx >= img1.cols - half_patch_size ||
                kp.pt.y + dy <= half_patch_size || kp.pt.y + dy >= img1.rows - half_patch_size)
            {   // go outside
                succ = false;
                break;
            }

            // compute cost and jacobian
            for (int x = -half_patch_size; x < half_patch_size; x++)
                for (int y = -half_patch_size; y < half_patch_size; y++)
                {

                    // TODO START YOUR CODE HERE (~8 lines)
                    double error = 0;
                    Eigen::RowVector2d J;  // Jacobian
                    if(kp.pt.x+x+dx-1>=0&&kp.pt.x+x+dx<=img2.cols&&kp.pt.y+y+dy-1>=0&&kp.pt.y+y+dy<=img2.rows)
                    {
                        if (inverse == false)
                        {
                            // Forward Jacobian
                            J(0,0)=(GetPixelValue(img2,kp.pt.x+x+dx+0.5,kp.pt.y+y+dy)-GetPixelValue(img2,kp.pt.x+dx-0.5,kp.pt.y+dy))/2;
                            J(0,1)=(GetPixelValue(img2,kp.pt.x+x+dx,kp.pt.y+y+dy+0.5)-GetPixelValue(img2,kp.pt.x+dx,kp.pt.y+dy-0.5))/2;
                            //cout<<J(0,0)<<","<<J(0,1)<<endl;
                        } else
                        {
                            // Inverse Jacobian
                            // NOTE this J does not change when dx, dy is updated, so we can store it and only compute error
                            J(0,0)=(GetPixelValue(img1,kp.pt.x+x+0.5,kp.pt.y+y)-GetPixelValue(img1,kp.pt.x-0.5,kp.pt.y))/2;
                            J(0,1)=(GetPixelValue(img1,kp.pt.x+x,kp.pt.y+y+0.5)-GetPixelValue(img1,kp.pt.x,kp.pt.y-0.5))/2;
                        }
                        // compute H, b and set cost;
                        error=(GetPixelValue(img2,kp.pt.x+x+dx,kp.pt.y+y+dy)-GetPixelValue(img1,kp.pt.x+x,kp.pt.y+y))+J(0,0)*dx+J(0,1)*dy;
                        H+=J.transpose()*J;
                        b+=-J.transpose()*error;
                        cost+=error*error;
                    } else
                        continue;

                    // TODO END YOUR CODE HERE
                }

            // compute update
            // TODO START YOUR CODE HERE (~1 lines)
            Eigen::Vector2d update;
            update=H.colPivHouseholderQr().solve(b);
            //cout<<update(0,0)<<","<<update(1,0)<<endl;
            // TODO END YOUR CODE HERE

            if (isnan(update[0])) {
                // sometimes occurred when we have a black or white patch and H is irreversible
                cout << "update is nan" << endl;
                succ = false;
                break;
            }
            if (iter > 0 && cost > lastCost)
            {
                cout << "cost increased: " << cost << ", " << lastCost << endl;
                break;
            }

            // update dx, dy
            dx += update[0];
            dy += update[1];
            lastCost = cost;
            succ = true;
        }

        success.push_back(succ);

        // set kp2
        if (have_initial) {
            kp2[i].pt = kp.pt + Point2f(dx, dy);
        } else {
            KeyPoint tracked = kp;
            tracked.pt += cv::Point2f(dx, dy);
            kp2.push_back(tracked);
        }
    }
}

void OpticalFlowMultiLevel(
        const Mat &img1,
        const Mat &img2,
        const vector<KeyPoint> &kp1,
        vector<KeyPoint> &kp2,
        vector<bool> &success,
        bool inverse) {

    // parameters
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    // create pyramids
    vector<Mat> pyr1, pyr2; // image pyramids
    // TODO START YOUR CODE HERE (~8 lines)
    Mat img(img1.rows,img1.cols,CV_8UC1);
    for (int i = 0; i < pyramids; i++)
    {
        //求缩放的图像大小
        double k=scales[i];
        pyr1.push_back(img);
        pyr1[i].cols=(int)(img1.cols*k);
        pyr1[i].rows=(int)(img1.rows*k);
        //求缩放的图像对应原图像点的像素值
        for(int n=0;n<pyr1[i].rows;n++)
            for(int j=0;j<pyr1[i].cols;j++)
            {
                pyr1[i].at<uchar>(n,j)=GetPixelValue(img1,j/k,n/k);
            }
        pyr2.push_back(img);
        pyr2[i].cols=(int)(img2.cols*k);
        pyr2[i].rows=(int)(img2.rows*k);
        //求缩放的图像对应原图像点的像素值
        for(int n=0;n<pyr2[i].rows;n++)
            for(int j=0;j<pyr2[i].cols;j++)
            {
                pyr2[i].at<uchar>(n,j)=GetPixelValue(img2,j/k,n/k);
            }
    }
    // TODO END YOUR CODE HERE

    // coarse-to-fine LK tracking in pyramids
    // TODO START YOUR CODE HERE
    //图像一缩放4层后的关键点坐标
    vector<KeyPoint> kp=kp1,kp0=kp1,kp20;
    double k=scales[3];
    //将img1的图像中的关键点的坐标进行缩放，得到缩放后图像关键点的位置
    for(int p=0;p<kp1.size();p++)
    {
        kp[p].pt.x=kp0[p].pt.x*k;
        kp[p].pt.y=kp0[p].pt.y*k;
    }
    //图像一缩放后的图像的单层光流投射到图像二中的单层光流中
    OpticalFlowSingleLevel(pyr1[3],pyr2[3],kp,kp20,success,inverse);
    //将图像二的关键点投射到下一层
    for(int w=3;w>0;w--)
    {
        for(auto &u:kp20)//图二每一层都要投射到下一层
        {
            u.pt.x*=2;
            u.pt.y*=2;
        }
        for(auto &v:kp)//图一每一层投射到下一层
        {
            v.pt.x*=2;
            v.pt.y*=2;
        }
        OpticalFlowSingleLevel(pyr1[w-1],pyr2[w-1],kp,kp20,success, inverse);//图一的每一层对图二的每一层做光流
    }
    //OpticalFlowSingleLevel(img1,img2,kp,kp20,success);
    kp2=kp20;
    // TODO END YOUR CODE HERE
    // don't forget to set the results into kp2
}