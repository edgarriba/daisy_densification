#ifndef ROBUST_MATCHER_H_
#define ROBUST_MATCHER_H_

#include <iostream>

#include <opencv2/core/version.hpp>
#include <opencv2/opencv_modules.hpp>

#ifndef CV_VERSION_EPOCH
#  include <opencv2/core.hpp>
#  include <opencv2/imgproc.hpp>
#  include <opencv2/highgui.hpp>
#  include <opencv2/features2d.hpp>
#  include <opencv2/calib3d.hpp>
#  include <opencv2/viz.hpp>
#else
#  include <opencv2/core/core.hpp>
#  include <opencv2/imgproc/imgproc.hpp>
#  include <opencv2/highgui/highgui.hpp>
#  include <opencv2/features2d/features2d.hpp>
#  include <opencv2/calib3d/calib3d.hpp>
#  include <opencv2/viz.hpp>
#endif

#ifdef HAVE_OPENCV_XFEATURES2D
#  include <opencv2/xfeatures2d.hpp>
#endif

class RobustMatcher {
public:

  // Constructor (Specify a descriptor type)
  explicit RobustMatcher(const cv::Ptr<cv::Feature2D> detector,
                         const cv::Ptr<cv::Feature2D> describer);

  bool computeDenseDepth (const cv::Mat& img1,
                          const cv::Mat& img2,
                          const cv::Matx33d& K1,
                          const cv::Matx33d& K2,
                          const cv::Matx33d& R,
                          const cv::Vec3d& t,
                          cv::Mat& points3d);

private:

  void rectifyStereoPair(const cv::Mat& img1,
                         const cv::Mat& img2,
                         const cv::Matx33d& K1,
                         const cv::Matx33d& K2,
                         const cv::Matx33d& R,
                         const cv::Vec3d& t,
                         cv::Matx33d& R1,
                         cv::Matx33d& R2,
                         cv::Mat& img1r,
                         cv::Mat& img2r,
                         cv::Mat& Q);

  void computeDisparity(const cv::Mat& img1,
                        const cv::Mat& img2,
                        const cv::Matx33d& R1,
                        const cv::Matx33d& R2,
                        cv::Mat& disparity_map);

  void
  computePointcloudFromDisparity(const cv::Mat& disparity_map,
                                 const cv::Mat& Q,
                                 cv::Mat& points3d);

  bool isDepthValid(const cv::Mat_<double>& epiline,
                    const int& _desc_radius,
                    int& _maxLoc);

  cv::Ptr<cv::Feature2D> detector_;
  cv::Ptr<cv::Feature2D> describer_;
};

#endif // ROBUST_MATCHER_H_

