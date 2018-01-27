#include <sensor_msgs/CameraInfo.h>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <image_geometry/pinhole_camera_model.h>

namespace custom_landmark_2d {

class Frame {
  public:
    cv::Point p1; // lower-left point of the frame
    cv::Point p2; // upper-right point of the frame

    float score; // the score of the current frame

    Frame();
    Frame(const cv::Point p1, const cv::Point p2, const float score);
};

class Matcher {

  public:
      int count_times; 		   // #times of scaling in each direction
      float raw_match_limit; // the threshold for BEST acceptable matching points
      float match_limit;     // the threshold for ALL acceptable matching points

      Matcher();

      void set_template(const cv::Mat& templ);
      void set_cam_model(const sensor_msgs::CameraInfoConstPtr& camera_info);

      // takes in an output parameter that contains all frames of matched objects in the scene;
      // returns true if there is at least one matched frame, and false otherwise
      bool match(const cv::Mat& scene, std::vector<Frame>& lst);
      // outputs each matched object as a single point cloud, in a vector of point cloud pointers;
      // depth MUST be the registered depth image of rgb
      bool match_clouds(const cv::Mat& rgb, const cv::Mat& depth, 
                        std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& object_clouds);
      // given a pre-computed vector of matched frames in the input rgb & depth scene,
      // outputs each matched object as a single point cloud, in a vector of point cloud pointers
      bool match_clouds_wframes(const cv::Mat& rgb, const cv::Mat& depth, std::vector<Frame>& lst,
                                std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& object_clouds);

  private:
      int match_method;
      int x_dist, y_dist;
      cv::Mat templ;
      image_geometry::PinholeCameraModel cam_model;

      // performs a single match on the given scene & templ, returns the max match_score	
      double exact_match(const cv::Mat& scene, const cv::Mat& templ, std::vector<Frame>& matching);
      // checks whether point(x, y) is around any frame in the vector, returns such frame if found
      bool around_frame(int x, int y, std::vector<Frame>& matching, Frame** p_ptr_ptr);
      // checks whether point(x, y) is around point p using the current x_dist & y_dist
      bool around_point(int x, int y, cv::Point& p);
};

}  // namespace custom_landmark_2d