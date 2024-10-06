#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define _USE_MATH_DEFINES
double pi = M_PI;

struct point_distance
{
    Point2f point;
    int distance;
    point_distance(Point2f point, int distance) : point(point), distance(distance) {}
};

double calculate_distance(const cv::Point2f &pt1, const cv::Point2f &pt2);
double calculate_limit_angle(const Point2f &center, const Point2f &p);
double calculate_angle(const Point2f &center, const Point2f &p);

void split_img_to_masks(Mat img, Mat &r_mask, Mat &b_mask);
vector<RotatedRect> find_rect_lights(Mat &input_frame, Mat &mask, bool isPutText);
bool is_light_pairs(RotatedRect rect1, RotatedRect rect2, Mat &input_frame);
vector<Point2f> find_armor(RotatedRect rect1, RotatedRect rect2, Mat input_frame);
bool is_armor(vector<Point2f> points,Mat &input_frame);
vector<vector<Point2f>> find_armors(vector<RotatedRect> rects, Mat input_frame);
void sort_points(vector<Point2f> &points, Point2f &center);
void solve_pose_and_mat(vector<Point2f> points, Mat input_frame);
