#include <sensor_msgs/CameraInfo.h>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <image_geometry/pinhole_camera_model.h>

namespace custom_landmark_2d {

class Frame {
    public:
        cv::Point p1; // upper-left point of the frame
        cv::Point p2; // lower-right point of the frame

        float score; // the score of the current frame

        Frame();
        Frame(const cv::Point p1, const cv::Point p2, const float score);
};

class Matcher {

    public:
        int count_times; 		   // #times of scaling in each direction
        float match_limit;     // the threshold for acceptable matching points

        Matcher();

        void set_template(const cv::Mat& templ);
        void set_cam_model(const sensor_msgs::CameraInfoConstPtr& camera_info);

        // takes in an output parameter that contains all frames of matched objects in the scene;
        // returns true if there is at least one matched frame, and false otherwise
        bool Match(const cv::Mat& scene, std::vector<Frame>* lst);

        // outputs each matched object as a single point cloud, in a vector of point cloud pointers;
        // depth MUST be the registered depth image of rgb
        // returns true if there is at least one matched object cloud, and false otherwise
        bool Match(const cv::Mat& rgb, const cv::Mat& depth, 
                   std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>* object_clouds);

        // given a single pre-computed matched frame in the input rgb & depth scene,
        // output the single matched object as a point cloud
        // returns true if object_cloud is not empty
        bool FrameToCloud(const cv::Mat& rgb, const cv::Mat& depth, const Frame& f,
                          pcl::PointCloud<pcl::PointXYZRGB>::Ptr object_cloud);

    private:
        int match_method_;
        int x_dist_, y_dist_;
        cv::Mat templ_;
        image_geometry::PinholeCameraModel cam_model_;

        // performs a single match on the given scene & scaled_templ
        void exact_match(const cv::Mat& scene, const cv::Mat& scaled_templ, std::vector<Frame>* lst);
        // checks whether point(x, y) is around any frame in the vector, returns such frame if found
        bool around_frame(int x, int y, std::vector<Frame>* lst, Frame** p_ptr_ptr);
        // checks whether point(x, y) is around point p using the current x_dist_ & y_dist_
        bool around_point(int x, int y, cv::Point& p);
};

}  // namespace custom_landmark_2d