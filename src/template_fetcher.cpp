#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <sstream>

using namespace std;
using namespace cv;

void fetcher(const sensor_msgs::Image::ConstPtr& msg);

// run this class like this:
// rosrun custom_landmark_2d template_fetcher rgb:=/head_camera/rgb/image_raw
//
// it saves a snapshot of the image to the current directory.
int main( int argc, char** argv ) {

  ros::init(argc, argv, "template_fetcher");

  ros::NodeHandle n;

  ros::Subscriber sub = n.subscribe("rgb", 5, fetcher); // rgb:=/head_camera/rgb/image_raw


  ros::spin();


  return 0;
}

void fetcher(const sensor_msgs::Image::ConstPtr& msg) {
  if (ros::ok()) {
    cv_bridge::CvImagePtr cv_ptr;

    try {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8); // , sensor_msgs::image_encodings::RGB8
    } 
    catch (cv_bridge::Exception& e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      exit(1);
    }

    imwrite("template.jpg", cv_ptr->image);
    ROS_INFO("image captured successfully, shutting down...");
    
    ros::shutdown();
  }
}