#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <iostream>

using namespace std;
using namespace cv;

void fetcher(const sensor_msgs::Image::ConstPtr& msg);

void print_usage() {
  std::cout << "\n"
            << "Saves a snapshot of image from the input sensor_msgs::Image topic "
            << "to the current directory.\n" 
            << "\n"
            << "Usage: rosrun custom_landmark_2d template_fetcher rgb:=/topic1" 
            << std::endl;
}

// run this class like this:
// rosrun custom_landmark_2d template_fetcher rgb:=/head_mount_kinect/rgb/image_raw
int main( int argc, char** argv ) {

  if (argc != 2) {
    print_usage();
    return 1;
  }

  ros::init(argc, argv, "template_fetcher");

  ros::NodeHandle n;
  ros::Subscriber sub = n.subscribe("rgb", 5, fetcher);

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