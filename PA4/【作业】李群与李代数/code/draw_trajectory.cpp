#include <sophus/se3.h>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>

// need pangolin for plotting trajectory
#include <pangolin/pangolin.h>

using namespace std;
using namespace Eigen;
// path to trajectory file
string trajectory_file = "/home/liuhongwei/workspace/slam/PA4/【作业】李群与李代数/code/trajectory.txt";
//string s="/home/liuhongwei/workspace/slam/PA3/【作业】李群与李代数/code/groundtruth.txt";
//读取估计轨迹
ifstream isr("/home/liuhongwei/workspace/slam/PA3/【作业】李群与李代数/code/estimated.txt");

// function for plotting trajectory, don't edit this code
// start point is red and end point is blue
void DrawTrajectory(vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>>);

int main(int argc, char **argv) {

    vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses1;
    //vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses2;
    /// implement pose reading code
    // start your code here (5~10 lines)
    ifstream it1(trajectory_file);
    string line1,line2;
    while(getline(it1,line1))
    {
        string n1,n2,n3,n4,t1,t2,t3;
        istringstream is1(line1);
        is1>>n1;
        is1>>t1;is1>>t2;is1>>t3;
        is1>>n1;is1>>n2;is1>>n3;is1>>n4;
        Vector3d t(stod(t1),stod(t2),stod(t3));
        Eigen::Quaterniond q1(stod(n4),stod(n1),stod(n2),stod(n3));
        Sophus::SE3 T1(q1,t);
        poses1.push_back(T1);
    }
    // end your code here

    // draw trajectory in pangolin
    DrawTrajectory(poses1);
    return 0;
}

/*******************************************************************************************/
void DrawTrajectory(vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses) {
    if (poses.empty()) {
        cerr << "Trajectory is empty!" << endl;
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
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));


    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glLineWidth(2);
        for (size_t i = 0; i < poses.size() - 1; i++) {
            glColor3f(1 - (float) i / poses.size(), 0.0f, (float) i / poses.size());
            glBegin(GL_LINES);
            auto p1 = poses[i], p2 = poses[i + 1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }

}