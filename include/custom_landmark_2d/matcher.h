#include <sensor_msgs/CameraInfo.h>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <image_geometry/pinhole_camera_model.h>

namespace custom_landmark_2d {

/// \brief Represents a single instance of matched object in the 2D rgb scene.
class Frame {
    public:
        /// \brief The upper-left point of this frame (x_min, y_min)
        cv::Point p1;
        /// \brief The lower-right point of this frame (x_max, y_max)
        cv::Point p2;
        /// \brief The score of this frame
        float score; 

        Frame();
        Frame(const cv::Point p1, const cv::Point p2, const float score);
};

/// \brief The main API for 2D object recognition and projection to 3D pointclouds.
///
/// It finds all the matched objects (each represented as a 'frame' by custom_landmark_2d::Frame) in the given rgb image.
/// Given an additional registered depth image, it can also project the matched objects into 3D pointclouds.
///
/// \par API Usage Example:
///
/// \code
///   custom_landmark_2d::Matcher matcher;
///
///   matcher.set_template(templ);
///   matcher.set_cam_model(camera_info);
///
///   // this is the threshold for ALL acceptable matching frames,
///   // with value ranging from 0.0 to 1.0(perfect match). The default value is 0.68. 
///   // this value is likely needed to be adjusted for optimal results on different objects.
///   matcher.match_limit = 0.6;
///
///
///   // we want to find out how many matched objects are in the rgb image:
///   std::vector<custom_landmark_2d::Frame> lst;
///   if (matcher.Match(rgb_image, &lst)) {
///     // we get >=1 matched frames/objects
///   }
///
///
///   // we now want the pointcloud containing only the second matched object:
///   custom_landmark_2d::Frame object_frame = lst[1];
///   pcl::PointCloud<pcl::PointXYZRGB>::Ptr object_cloud;
///   matcher.FrameToCloud(rgb_image, depth_image, object_frame, object_cloud);
///
///
///   // we don't care about 2D points;
///   // just directly give us all matched objects, each in a pointcloud:
///   std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> object_clouds;
///   if (matcher.Match(rgb_image, depth_image, &object_cloud)) {
///     // we get >=1 matched objects
///   }
/// \endcode
///
class Matcher {

    public:
        /// \brief #times of scaling in each direction
        ///
        /// The default value is 2, which means a total scaling range of (0.8 ~ 1.2) * original_scale
        int count_times;
        /// \brief The threshold for ALL acceptable matching frames, with value ranging from 0.0 to 1.0(perfect match).
        ///
        /// The default value is 0.68. This value is likely needed to be adjusted for optimal results on different objects.
        float match_limit;

        /// \brief The default constructors
        ///
        /// Sets all public variables to their default values.
        Matcher();

        /// \brief Sets the template image (the object to be recognized)
        void set_template(const cv::Mat& templ);

        /// \brief Sets the camera model
        ///
        /// This is only needed for projection to 3D pointclouds.
        void set_cam_model(const sensor_msgs::CameraInfoConstPtr& camera_info);

        /// \brief Performs matching and outputs each matched object as a Frame.
        /// 
        /// Returns true if there is at least one matched object, and false otherwise.
        ///
        /// \param[in] scene The 2D rgb scene within which to find matched objects
        ///
        /// \param[out] lst The vector that contains all frames of matched objects in the scene
        bool Match(const cv::Mat& scene, std::vector<Frame>* lst);

        /// \brief Performs matching and outputs each matched object as a pcl pointcloud.
        ///        Currently only supports depth image type of <b>CV_32FC1</b>.
        /// 
        /// Returns true if there is at least one matched object, and false otherwise.
        /// This method asserts that the rgb and depth image have the same dimension and
        /// the depth image is of type CV_32FC1.
        ///
        /// \param[in] rgb The 2D rgb scene within which to find matched objects
        /// \param[in] depth The 2D <b>registered</b> depth image corresponds to the rgb scene 
        ///                  (must have the same dimension as the rgb image)
        ///
        /// \param[out] lst The vector of matched objects in the rgb scene, each as a pcl pointcloud
        bool Match(const cv::Mat& rgb, const cv::Mat& depth, 
                   std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>* object_clouds);

        /// \brief Given a single pre-computed matched Frame, projects it into a pcl pointcloud.
        ///        Currently only supports depth image type of <b>CV_32FC1</b>.
        /// 
        /// Returns true if object_cloud is not empty.
        /// This method asserts that the rgb and depth image have the same dimension and
        /// the depth image is of type CV_32FC1.
        ///
        /// \param[in] rgb The 2D rgb scene within which we found the input matched Frame f
        /// \param[in] depth The 2D <b>registered</b> depth image corresponds to the rgb scene 
        ///                  (must have the same dimension as the rgb image)
        /// \param[in] f A single pre-computed matched Frame in the input rgb scene
        ///
        /// \param[out] object_cloud The pcl pointcloud of the matched object represented by the input Frame f
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
        // check that rgb & depth have the required properties like dimensions & encodings
        void check_rgbd(const cv::Mat& rgb, const cv::Mat& depth);
};

}  // namespace custom_landmark_2d