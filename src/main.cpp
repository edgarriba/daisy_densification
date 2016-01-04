#include <iostream>

#include "robust_matcher.h"

using namespace std;
using namespace cv;

int main( int argc, char** argv )
{
    cout << "Hello Daisy" << endl;

    if (argc < 2) {
        cout << "[ERROR] Path to data not provided." << endl;
        cout << "Usage ./main [<data_dir>]" << endl;
        return 0;
    }

    const Ptr<Feature2D> descriptor = xfeatures2d::DAISY::create(
        15, 3, 8, 8, xfeatures2d::DAISY::NRM_FULL);

    RobustMatcher matcher(descriptor, descriptor);

    string data_dir = string(argv[1]),
           img1_dir = data_dir + string("temple0001.png"),
           img2_dir = data_dir + string("temple0002.png");

    Mat img1 = imread(img1_dir, 0),
        img2 = imread(img2_dir, 0);

    // Set scene calibration

    Matx33d K1(1520.400000,    0.000000, 302.320000,
                  0.000000, 1525.900000, 246.870000,
                  0.000000,    0.000000,   1.000000);

    Matx33d K2(1520.400000,    0.000000, 302.320000,
                  0.000000, 1525.900000, 246.870000,
                  0.000000,    0.000000,   1.000000);

    Matx33d R1( 0.01551372092999463200,  0.99884343581246959000, -0.04550950666890610900,
                0.99922238739871228000, -0.01713749902859566800, -0.03550952897832390700,
               -0.03624837905512174500, -0.04492323298011671700, -0.99833258894743582000);

    Vec3d t1(-0.05998547900141842900, 0.00400788029504099870, 0.57088647431543438000);

    Matx33d R2( 0.01614490437974924100,  0.99884677638989772000, -0.04521569813768747100,
                0.99856922398083869000, -0.01380176413826810800,  0.05166252244109931900,
                0.05097888759941932700, -0.04598509108593063600, -0.99764047853770654000);

    Vec3d t2(-0.05998004112555456500, 0.00374555199382083440, 0.57175508950314680000);

    // Compute relative motion

    Mat t;
    Mat R;

    R = Mat(R2) * Mat(R1).t();
    t = Mat(t2) - Mat(R) * Mat(t1);

    // Compute dense pointclou

    Mat points3d;
    matcher.computeDenseDepth(img1, img2, K1, K2, R, t, points3d);

    return 0;
}


