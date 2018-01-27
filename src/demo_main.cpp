#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>
#include <time.h>

#include <custom_landmark_2d/matcher.h>

typedef pcl::PointXYZRGB PointC;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudC;

using namespace sensor_msgs;
using namespace message_filters;

namespace custom_landmark_2d {

class Demor {
	public:
		Matcher matcher;
		sensor_msgs::CameraInfoConstPtr camera_info;

		ros::Publisher cloud_pub;
		ros::Publisher matched_scene_pub;

		void callback(const sensor_msgs::ImageConstPtr& rgb, const sensor_msgs::ImageConstPtr& depth);
};

void Demor::callback(const sensor_msgs::ImageConstPtr& rgb, const sensor_msgs::ImageConstPtr& depth) {

  // convert sensor_msgs::Images to cv::Mats
  cv_bridge::CvImagePtr rgb_ptr;
  cv_bridge::CvImagePtr depth_ptr;

  try {
    rgb_ptr = cv_bridge::toCvCopy(rgb, sensor_msgs::image_encodings::BGR8);
    depth_ptr = cv_bridge::toCvCopy(depth); // 32FC1
  } 
  catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    ros::shutdown(); 
  }

  // run matching on the image
  std::vector<custom_landmark_2d::Frame> lst;

  float start_tick = clock();
  bool result = matcher.match(rgb_ptr->image, lst);
  float end_tick = clock();
  ROS_INFO("match() runtime: %f", (end_tick - start_tick) / CLOCKS_PER_SEC); 

  if (!result) {
    ROS_INFO("no matched objects in 2d rgb scene...\n");
    return; 
  }

  // the result pointCloud vector of matched objects
  std::vector<PointCloudC::Ptr> object_clouds;

  start_tick = clock();
  result = matcher.match_clouds_wframes(rgb_ptr->image, depth_ptr->image, lst, object_clouds);
  end_tick = clock();
  ROS_INFO("match_clouds_wframes() runtime: %f\n", (end_tick - start_tick) / CLOCKS_PER_SEC); 

  ROS_INFO("#matched 2D objects: %lu\n", lst.size()); 
  ROS_INFO("-----------");
  for (int i = 0; i < lst.size(); i++) {
    Frame& f = lst[i];
    // annotates matched parts on rgb scene
    std::ostringstream stm;
    stm << i;
    rectangle( rgb_ptr->image, f.p1, f.p2, cv::Scalar(255, 255, 0), 5, 8, 0 );
    putText(rgb_ptr->image, stm.str(), cv::Point(f.p1.x - 10, f.p1.y - 10), 
            cv::FONT_HERSHEY_COMPLEX_SMALL, 1.5, cv::Scalar(255, 255, 0), 2, CV_AA);
    ROS_INFO("frame score: %f, p1 pos: [%d, %d], index: %d", f.score, f.p1.x, f.p1.y, i);
  }
  ROS_INFO("-----------\n");

  matched_scene_pub.publish(rgb_ptr->toImageMsg());

  if (!result) {
    ROS_INFO("no valid matched object coordinate...\n");
    return; 
  }

  // the single pointCloud for better display 
  PointCloudC::Ptr pcl_cloud(new PointCloudC());

  // concatenate the vector of pointClouds
  for (std::vector<PointCloudC::Ptr>::iterator it = object_clouds.begin(); it != object_clouds.end(); it++) {
    *pcl_cloud += **it;
  }
  ROS_INFO("#matched 3D objects: %lu", object_clouds.size()); 
  ROS_INFO("pcl cloud size: %lu\n", pcl_cloud->size());  

  sensor_msgs::PointCloud2 ros_cloud;
  pcl::toROSMsg(*pcl_cloud, ros_cloud);
  ros_cloud.header.frame_id = camera_info->header.frame_id; // head_camera_rgb_optical_frame

  cloud_pub.publish(ros_cloud);
}

}

// run this class like this:
// rosrun custom_landmark_2d demo template.jpg rgb:=/head_camera/rgb/image_raw depth:=/head_camera/depth_registered/image_raw
int main(int argc, char** argv) {

  ros::init(argc, argv, "demo");
  ros::NodeHandle nh;

  // Load template from stdin
  cv::Mat templ = cv::imread( argv[1], 1 );

  if ( !templ.data ) {
      ROS_ERROR("No template data \n");
      return -1;
  }

  custom_landmark_2d::Demor demor;

  // fetch CameraInfo
  demor.camera_info = ros::topic::waitForMessage<sensor_msgs::CameraInfo>("/head_camera/rgb/camera_info");

  ROS_INFO("received camear_info...");  

  // setup the matcher
  demor.matcher.set_template(templ);
  demor.matcher.set_cam_model(demor.camera_info);

  // setup published topics
  demor.cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("generated_cloud", 1, true);
  demor.matched_scene_pub = nh.advertise<sensor_msgs::Image>("image_out", 100);

  message_filters::Subscriber<Image> rgb_sub(nh, "rgb", 1); // rgb:=/head_camera/rgb/image_raw
  message_filters::Subscriber<Image> depth_sub(nh, "depth", 1); // depth:=/head_camera/depth_registered/image_raw

  typedef sync_policies::ApproximateTime<Image, Image> SyncPolicy;
  // ApproximateTime takes a queue size as its constructor argument, hence SyncPolicy(10)
  Synchronizer<SyncPolicy> sync(SyncPolicy(10), rgb_sub, depth_sub);
  sync.registerCallback(boost::bind(&custom_landmark_2d::Demor::callback, &demor, _1, _2));
  
  ros::spin();

  return 0;
}