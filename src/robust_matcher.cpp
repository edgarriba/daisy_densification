#include "robust_matcher.h"

using namespace std;
using namespace cv;

RobustMatcher::RobustMatcher(
    const Ptr<Feature2D> detector,
    const Ptr<Feature2D> describer) {
  detector_ = detector;
  describer_ = describer;
}

bool
RobustMatcher::computeDenseDepth(const Mat& img1,
                                 const Mat& img2,
                                 const Matx33d& K1,
                                 const Matx33d& K2,
                                 const Matx33d& R,
                                 const Vec3d& t,
                                 Mat& points3d) {
  //-- 1. Rectify images

  Matx33d R1, R2;
  Mat img1r, img2r, Q;
  rectifyStereoPair(img1, img2, K1, K2, R, t, R1, R2, img1r, img2r, Q);


  //-- 2. Compute disparity between stereo pair

  Mat disparity;
  computeDisparity(img1r, img2r, R1, R2, disparity);


  // -- 3. Compute 3d points from disparity and Q
  //computePointcloudFromDisparity(disparity, Q, points3d);

  return true;
}


void
RobustMatcher::computePointcloudFromDisparity(const Mat& disparity_map,
                                              const Mat& Q,
                                              Mat& points3d) {
  // OpenCV function to compute pointcloud
  reprojectImageTo3D(disparity_map, points3d, Q, false, -1);

  /// Plot 3d points with viz3d

  viz::Viz3d my_window("Local dense reconstuction");
             my_window.setBackgroundColor(); // black by default

  const double  f = Q.at<double>(2,3),
               cx = Q.at<double>(0,3),
               cy = Q.at<double>(1,3),
               base = Q.at<double>(3,2);

  cout << "f: " << f << endl;
  cout << "cx: " << -cx << endl;
  cout << "cy: " << -cy << endl;
  cout << "base: " << base << endl;

  vector<Vec3f> point_cloud;
  for (int i = 0; i < points3d.rows; ++i) {
    for (int j = 0; j < points3d.cols; ++j) {
      // recover computed points3d
      cv::Vec3f point3d = points3d.at<Vec3f>(i,j);
      if ( std::isfinite(point3d[0]) &&
           std::isfinite(point3d[1]) &&
           std::isfinite(point3d[2]) )
        point_cloud.push_back(point3d);

      //if (point_cloud.size() > 0) cout << point_cloud.back() << endl;
    }
  }

  cout << "Extracted " << point_cloud.size() << endl;

  viz::WCloud cloud_widget(point_cloud, viz::Color::green());

  my_window.showWidget("point_cloud", cloud_widget);
  my_window.spin();

}

void
RobustMatcher::rectifyStereoPair(const Mat& img1,
                                 const Mat& img2,
                                 const Matx33d& K1,
                                 const Matx33d& K2,
                                 const Matx33d& R,
                                 const Vec3d& t,
                                 Matx33d& _R1,
                                 Matx33d& _R2,
                                 Mat& img1r,
                                 Mat& img2r,
                                 Mat& _Q) {
  Matx33d M1 = K1, M2 = K2;
  Vec4d D1 = Vec4d(0,0,0,0), D2 = Vec4d(0,0,0,0);

  Rect roi1, roi2;
  Mat Q;

  Mat R1, P1, R2, P2;

  int scale = 4;
  Size img_size = img1.size(),
       new_size = Size(scale*img_size.height, scale*img_size.width);

  stereoRectify(M1, D1, M2, D2, img_size, R, t, R1, R2, P1, P2, Q,
                CALIB_ZERO_DISPARITY, -1, new_size, &roi1, &roi2);

  Mat map11, map12, map21, map22;
  initUndistortRectifyMap(M1, D1, R1, P1, new_size, CV_16SC2, map11, map12);
  initUndistortRectifyMap(M2, D2, R2, P2, new_size, CV_16SC2, map21, map22);

  remap(img1, img1r, map11, map12, INTER_LINEAR);
  remap(img2, img2r, map21, map22, INTER_LINEAR);

  // crop by hand
  img1r = img1r(roi1);
  img2r = img2r(roi2);

  // hack Q matrix
  Q.at<double>(0,3) = -K1(0,2);
  Q.at<double>(1,3) = -K1(1,2);

  // set Q matrix to compute 3d points
  _Q = Q;
  _R1 = R1;
  _R2 = R2;

  // hack!
  resize(img2r, img2r, img1r.size());

  //imshow("Image Left",  img1r);
  //imshow("Image Right", img2r);
  //waitKey(0);

}


bool RobustMatcher::isDepthValid(const cv::Mat_<double>& epiline,
                                 const int& _desc_radius,
                                 int& _maxLoc) {

  const double disp_threshold = 0.8;

  Mat_<double> epiline_tmp = epiline.clone();

  // To decide whether or not to assign a depth to a pixel, we
  // look for the first two probability maxima along the uniformly
  // sampled epipolar line and consider the ratio of their values

  double minVal[2];
  double maxVal[2];
  Point minLoc[2];
  Point maxLoc[2];

  for (int i = 0; i < 2; ++i) {
    // find maximum value and location
    minMaxLoc( epiline_tmp, &minVal[i], &maxVal[i], &minLoc[i], &maxLoc[i] );

    // assign zero value in order to find the second best probability
    epiline_tmp(0,maxLoc[i].x) = 0.0;
  }

  // ratio between first and second maximum values
  double ratio = ( maxVal[0] != 0.0) ? maxVal[1] / maxVal[0] : 0.0;

  // distance between first and second values position
  int dist = std::abs(maxLoc[0].x - maxLoc[1].x);

  // return the maximum value position
  _maxLoc = maxLoc[0].x;

  // to assign a depth value:
  // 1. The ratio must be higher than a given threshold
  // 2. The ratio must be lower than 1.00 (0.999 is OK)
  // 3. The distance between max values must be lower than the descriptor radius
  return (//maxVal[0] >= disp_threshold &&
          ratio >= disp_threshold && ratio < 1 && dist <= _desc_radius);
}


void
RobustMatcher::computeDisparity(const Mat& img1,
                                const Mat& img2,
                                const Matx33d& R1,
                                const Matx33d& R2,
                                Mat& _disparity_map) {
  // Compute dense descriptors

  const int desc_radius = 15;
  const int desc_type = xfeatures2d::DAISY::NRM_FULL;

  Ptr<xfeatures2d::DAISY>
  daisy_left  = xfeatures2d::DAISY::create(desc_radius,3,8,8, desc_type);

  Ptr<xfeatures2d::DAISY>
  daisy_right  = xfeatures2d::DAISY::create(desc_radius,3,8,8, desc_type);

  Mat desc_l, desc_r;
  daisy_left->compute ( img1, desc_l );
  daisy_right->compute( img2, desc_r );


  // Used in order to filter some matches

  Mat img1_bin, img2_bin, mask;
  threshold(img1, img1_bin, 0, 255, THRESH_TRIANGLE);
  threshold(img2, img2_bin, 0, 255, THRESH_TRIANGLE);
  bitwise_or(img1_bin, img2_bin, mask);

  // Buffer disparity map
  Mat_<int> disparity_map(img1.size());

  // TODO: refine this parameters and find out how to use Z in the
  //       epipolar line.
  const double sigma = 1.0, Z = 1.0;
  const int n_rows = img1.rows;
  const int n_cols = img1.cols;
  const int max_disparity = 16;

  // Compute the probability of a pixel x having a depth d in one image

  // TODO: parallelise this for loop!

  for (int row = 0; row < n_rows; ++row) {

    cout << "\r *** PROGRESS " << (row+1)*100/n_rows << "% ***";

    // compute the probabilty distribution
    for (int pixel_l = 0; pixel_l < n_cols; ++pixel_l) {

      if ( !mask.at<uchar>(row, pixel_l) ) {
        disparity_map(row, pixel_l) = 0;
        continue;
      }

      // get left descriptor using daisy interface for pixel i
      vector<float> desc_left_vec(200, 0.0f);
      Mat_<float> desc_left(1, 200, &desc_left_vec[0]);

      daisy_left->GetDescriptor(row, pixel_l, 0, &desc_left_vec[0]);


      // container matrix to fill with the probability distribution
      Mat_<float> epiline(1, 2*max_disparity);

      // compute probability distribution in this epipolar line

      for ( int k = 0; k < 2*max_disparity; ++k) {

        double probability = 0.0;
        int pos = pixel_l - max_disparity + k;

        // Avoid margins

        if ( pos >= 0 && pos < n_cols ) {

          // take right descriptor to compare with pixel i descriptor
          // using daisy interface
          vector<float> desc_right_vec(200, 0.0f);
          Mat_<float> desc_right(1, 200, &desc_right_vec[0]);

          daisy_right->GetDescriptor(row, pos, 0, &desc_right_vec[0]);

          /*

          Probability formula in (1)

                        1            || D^i_x - D^j_x'(d) || ^2
                P(d) = --- exp ( - ------------------------------ )
                        Z                        σ

          D^i_x and D^j_x'(d) are the descriptors at x in one image and at the
                            corresponding point x'(d) in the other, assuming
                            that the depth is indeed d.

          The sharpness of the distribution is controlled by σ and Z is a normalizing
          constant that ensures that the probabilities sum to one.

          */

          // TODO: play with norm types

          //int norm_type = NORM_INF;
          //int norm_type = NORM_L1;
          int norm_type = NORM_L2;

          probability =
            ( 1.0 / Z ) * exp( -1.0 * pow(norm(desc_left - desc_right, norm_type), 2) / sigma );

          //cout << desc_left << endl;
          //cout << desc_right << endl;
          //cout << probability << endl;
          //cout << endl << "Pixel " << row << " " << pos << endl;
          //cin.get();
        }

        epiline(0, k) = probability;

      } // end_for k_search

      // Check if depth is valid in this epipolar line
      int maxLoc = 0, disparity = 0;

      if ( isDepthValid(epiline, desc_radius, maxLoc) ) {
        disparity = std::abs(max_disparity - maxLoc);

        Point2f p_left(row, pixel_l),
                p_right(row, pixel_l + maxLoc - max_disparity);

        //vec_correspondences_.push_back(make_pair(p_left, p_right));
      }

      disparity_map(row, pixel_l) = disparity;

    } // end_for pixel_l

  } // end_for row


  // Show disparity map

  Mat disparity_map_vis;
  normalize(disparity_map, disparity_map_vis, 0, 255, NORM_MINMAX, CV_8UC1);

  imshow("Disparity_map",  disparity_map_vis);
  waitKey(0);

  // Set output
  _disparity_map = disparity_map;
}