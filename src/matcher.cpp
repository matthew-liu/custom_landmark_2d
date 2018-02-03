#include <custom_landmark_2d/matcher.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sensor_msgs/CameraInfo.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <image_geometry/pinhole_camera_model.h>

#include <stdlib.h>
#include <assert.h> 

using namespace std;
using namespace cv;

typedef pcl::PointXYZRGB PointC;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudC;

namespace custom_landmark_2d {

Frame::Frame(const Point p1, const Point p2, const float score) : p1(p1), p2(p2), score(score) {}

Matcher::Matcher() : match_method_(CV_TM_CCOEFF_NORMED),
					 count_times(2),
					 match_limit(0.68) {}

void Matcher::set_template(const Mat& templ) {
	templ_ = templ;
}

void Matcher::set_cam_model(const sensor_msgs::CameraInfoConstPtr& camera_info) {
	cam_model_.fromCameraInfo(camera_info);
}

bool Matcher::Match(const Mat& rgb, const Mat& depth, vector<PointCloudC::Ptr>* object_clouds) {

	check_rgbd(rgb, depth);

	vector<Frame> lst;

    if (!Match(rgb, &lst)) return false; // no match found

    object_clouds->clear();
	
	PointCloudC::Ptr object_cloud;

	for (std::vector<Frame>::iterator f = lst.begin(); f != lst.end(); f++) {

		object_cloud = PointCloudC::Ptr(new PointCloudC());

		if (FrameToCloud(rgb, depth, *f, object_cloud)) {
			object_clouds->push_back(object_cloud);
		}
	}

	if (object_clouds->empty()) {
		return false;
	}
	return true;
}

bool Matcher::FrameToCloud(const Mat& rgb, const Mat& depth, const Frame& f, PointCloudC::Ptr object_cloud) {

	check_rgbd(rgb, depth);

	object_cloud->clear();

	for(int i = f.p1.y + 1; i < f.p2.y; i++) {
		for (int j = f.p1.x + 1; j < f.p2.x; j++) {

			float dist = depth.at<float>(i, j);

			if (!isnan(dist)) {
				cv::Vec3b color = rgb.at<cv::Vec3b>(i, j);
				cv::Point2d p_2d;
				p_2d.x = j;
				p_2d.y = i;

				cv::Point3d p_3d = cam_model_.projectPixelTo3dRay(p_2d);

				PointC pcl_point;

				pcl_point.x = p_3d.x * dist;
				pcl_point.y = p_3d.y * dist;
				pcl_point.z = p_3d.z * dist;

				// bgr
				pcl_point.b = static_cast<uint8_t> (color[0]);
				pcl_point.g = static_cast<uint8_t> (color[1]);
				pcl_point.r = static_cast<uint8_t> (color[2]);

				object_cloud->points.push_back(pcl_point);
			}
		}   
	}

	object_cloud->width = (int) object_cloud->points.size();
	object_cloud->height = 1;

	if (object_cloud->empty()) {
		return false;
	}
	return true;
}

void Matcher::check_rgbd(const Mat& rgb, const Mat& depth) {
	// currently only support depth image encoding of CV_32FC1
	assert(depth.type() == CV_32FC1);
	assert(rgb.cols == depth.cols && rgb.rows == depth.rows);
}

bool Matcher::Match(const Mat& scene, vector<Frame>* lst) {
	
	lst->clear();

	int counter = count_times;
	Mat scaled_templ;

	// scale up
	double factor = 1.0 + 0.1 * count_times;
	while (counter > 0) {

		resize(templ_, scaled_templ, Size(), factor, factor, INTER_LINEAR);
		exact_match(scene, scaled_templ, lst);

		factor -= 0.1;
		counter--;
	}

	// original scale
	exact_match(scene, templ_, lst);

	// scale down
	counter = count_times;
	factor  = 0.9;
	while (counter > 0) {

		resize(templ_, scaled_templ, Size(), factor, factor, INTER_AREA);
		exact_match(scene, scaled_templ, lst);

		factor -= 0.1;
		counter--;
	}

	if (lst->empty()) {
		return false;
	}

	return true;
}

// performs a single match on the given scene & scaled_templ
void Matcher::exact_match(const Mat& scene, const Mat& scaled_templ, vector<Frame>* lst) {

	x_dist_ = (int) scaled_templ.cols;
	y_dist_ = (int) scaled_templ.rows;

	Mat result; // the result matrix

	int result_cols = scene.cols - scaled_templ.cols + 1;
	int result_rows = scene.rows - scaled_templ.rows + 1;

	result.create(result_rows, result_cols, CV_32FC1);

	// Do the Matching
	matchTemplate(scene, scaled_templ, result, match_method_);

	// scan through result to find all matching points
	int i,j;
	float* p;
	for( i = 0; i < result.rows; i++) {
		p = result.ptr<float>(i);
		for ( j = 0; j < result.cols; j++) {
			if ( p[j] > match_limit) { // acceptable matching point
				Frame* f_ptr;
				if (around_frame(j, i, lst, &f_ptr)) {
					if (p[j] > f_ptr->score) { // current point has better score

						f_ptr->p1 = Point(j, i);
						f_ptr->p2 = Point(j + x_dist_, i + y_dist_);
            			f_ptr->score = p[j];
          			}
        		} else {
          			lst->push_back(Frame(Point(j, i), Point(j + x_dist_, i + y_dist_), p[j]));
        		}
      		}
  		}
	}
}

// checks whether point(x, y) is around any frame in the vector, returns such frame if found
bool Matcher::around_frame(int x, int y, vector<Frame>* lst, Frame** f_ptr_ptr) {
	if (lst->empty()) {
		return false;
	}

	for (vector<Frame>::iterator f = lst->begin(); f != lst->end(); f++) {
		if (around_point(x, y, f->p1)) {
 			*f_ptr_ptr = &(*f);
  			return true;
		}
	}
	return false;
}

// checks whether point(x, y) is around point p using the current x_dist_ & y_dist_
bool Matcher::around_point(int x, int y, Point& p) {
	if (abs(p.x - x) < x_dist_ && abs(p.y - y) < y_dist_) {
    	return true;
	}
  	return false;
}

}