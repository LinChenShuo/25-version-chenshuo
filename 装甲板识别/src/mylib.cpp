#include <iostream>
#include <opencv2/opencv.hpp>
#include "mylib.h"

using namespace std;
using namespace cv;

/**
 * @brief to_string函数的保留小数版本（方便自用查看输出来调参）
 *
 * @param 其他类型的输入值
 * @param 保留的小数点位数
 * @return 要打印的输出信息
 */

template <typename T>
string to_string_with_precision(const T a_value, const int n = 6)
{
    ostringstream out;
    out.precision(n);
    out << fixed << a_value;
    return out.str();
}

/**
 * @brief 获取两点间距离
 *
 * @param point1
 * @param point1
 * @return 距离
 */
double calculate_distance(const cv::Point2f &pt1, const cv::Point2f &pt2)
{
    return sqrt(pow(pt1.x - pt2.x, 2) + pow(pt1.y - pt2.y, 2));
}

/**
 * @brief 比较两点离某一中心点的角度（有个sort函数要用到，为了方便写出来了）
 *
 * @param center
 * @param point1
 * @param point2
 * @return 距离
 */
bool compare_angle(const Point2f &center, const Point2f &p1, const Point2f &p2)
{
    auto angle1 = calculate_angle(center, p1);
    auto angle2 = calculate_angle(center, p2);
    return angle1 > angle2;
}

/**
 * @brief 计算两点所成直线的角度（范围0到pi）
 *
 * @param point1
 * @param point2
 * @return 夹角
 */
double calculate_limit_angle(const Point2f &center, const Point2f &p)
{
    auto dx = p.x - center.x;
    auto dy = p.y - center.y;
    return atan(dy / dx);
}

/**
 * @brief 计算两点所成直线的角度（范围-pi到pi）
 *
 * @param point1
 * @param point2
 * @return 夹角
 */
double calculate_angle(const Point2f &center, const Point2f &p)
{
    auto dx = p.x - center.x;
    auto dy = p.y - center.y;
    return atan2(dy, dx);
}

/**
 * @brief 提取颜色
 *
 * @param 输入图像
 * @param 红色区域图（承载输出）
 * @param 蓝色区域图（承载输出）
 */
void split_img_to_masks(Mat img, Mat &r_mask, Mat &b_mask)
{
    Mat hsv_frame;
    Scalar hsv_b_l(90, 90, 140);
    Scalar hsv_b_h(140, 255, 255);
    Scalar hsv_r_l(0, 90, 150);
    Scalar hsv_r_h(35, 255, 255);
    cvtColor(img, hsv_frame, COLOR_BGR2HSV);
    inRange(hsv_frame, hsv_b_l, hsv_b_h, b_mask);
    inRange(hsv_frame, hsv_r_l, hsv_r_h, r_mask);
    // Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    // dilate(b_mask, b_mask, kernel);
    // dilate(r_mask, r_mask, kernel);
    // Mat kernel2 = getStructuringElement(MORPH_RECT, Size(3, 3));
    // erode(b_mask, b_mask, kernel2);
    // erode(r_mask, r_mask, kernel2);
    // Mat kernel1 = getStructuringElement(MORPH_RECT, Size(3, 3));
    // dilate(b_mask, b_mask, kernel1);
    // dilate(r_mask, r_mask, kernel1);
    GaussianBlur(b_mask, b_mask, Size(5, 5), 1);
    GaussianBlur(r_mask, r_mask, Size(5, 5), 1);
}

/**
 * @brief 找灯条
 *
 * @param 输入图像（用于putText显示信息）
 * @param 输入二值图（提取灯条用）
 * @param 是否显示信息（调试用）
 * @return 找到的灯条
 */
vector<RotatedRect> find_rect_lights(Mat &input_frame, Mat &mask, bool isPutText)
{
    vector<vector<Point>> contours;
    // 识别轮廓
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    vector<RotatedRect> rects;
    for (int i = 0; i < contours.size(); i++)
    {
        // vector<Point2f> r_contour_approx;
        // // 近似多边形轮廓，方便提取特征
        // approxPolyDP(contours.at(i), r_contour_approx, arcLength(contours.at(i), true) * 0.02, true);
        // 计算面积
        auto area = contourArea(contours.at(i));
        // 最小矩形轮廓，同样方便提取特征
        RotatedRect rect = minAreaRect(contours.at(i));
        Point2f points[4];
        rect.points(points);
        // putText(input_frame, "area=" + to_string_with_precision(area, 2), points[1], FONT_ITALIC, 0.5, Scalar(255, 255, 255));

        // 以下开始筛选
        // 面积限制
        if (area < 30)
        {
            if (isPutText)
                putText(input_frame, "area=" + to_string(round(area)), points[0], FONT_ITALIC, 0.5, Scalar(255, 255, 255));
            continue;
        }
        auto rect_area = rect.size.width * rect.size.height;
        auto area_pro = area / rect_area;
        // 通过面积比来估计是否为矩形进行筛选（讲真可能效果不好）
        if (area_pro < 0.3)
        {
            if (isPutText)
                putText(input_frame, "area_pro=" + to_string(area_pro) + "area=" + to_string(area) + "rect_area=" + to_string(rect_area), points[0], FONT_ITALIC, 0.5, Scalar(255, 255, 255));
            continue;
        }
        auto prop = max(rect.size.width, rect.size.height) / min(rect.size.width, rect.size.height);
        // 对应面积根据最小矩形的长宽比来筛选
        if (area < 100 and prop > 2)
        {
            if (isPutText)
                putText(input_frame, "prop=" + to_string(prop), points[3], FONT_ITALIC, 0.5, Scalar(255, 255, 255));
            continue;
        }
        else if (area < 200 and prop > 3.5)
        {
            if (isPutText)
                putText(input_frame, "prop=" + to_string(prop), points[3], FONT_ITALIC, 0.5, Scalar(255, 255, 255));
            continue;
        }
        else if (area >= 200 and prop > 5.5)
        {
            if (isPutText)
                putText(input_frame, "prop=" + to_string(prop), points[3], FONT_ITALIC, 0.5, Scalar(255, 255, 255));
            continue;
        }
        // // 画出近似矩形轮廓
        // for (int i = 0; i < 4; i++)
        // {
        //     line(input_frame, points[i], points[(i + 1) % 4], Scalar(255, 255, 255), 2);
        // }
        // 画出轮廓
        drawContours(input_frame, contours, i, Scalar(255, 255, 255), 2);
        if (isPutText)
        {
            putText(input_frame, "angle=" + to_string(rect.angle), points[3], FONT_ITALIC, 0.5, Scalar(255, 255, 255));
            putText(input_frame, "1", points[0], FONT_ITALIC, 0.5, Scalar(255, 255, 255));
            putText(input_frame, "2", points[1], FONT_ITALIC, 0.5, Scalar(255, 255, 255));
            putText(input_frame, "3", points[2], FONT_ITALIC, 0.5, Scalar(255, 255, 255));
            putText(input_frame, "4", points[3], FONT_ITALIC, 0.5, Scalar(255, 255, 255));
        }
        rects.push_back(rect);
    }
    return rects;
}

/**
 * @brief 筛选灯条对
 *
 * @param 灯条1
 * @param 灯条2
 * @param 输入图像（用于显示信息）
 * @return 是与否
 */
bool is_light_pairs(RotatedRect rect1, RotatedRect rect2, Mat &input_frame)
{
    bool flag = true;
    auto distance = calculate_distance(rect1.center, rect2.center);
    float distance_limit_h, distance_limit_l;
    auto area1 = rect1.size.height * rect1.size.width;
    auto area2 = rect2.size.height * rect2.size.width;
    auto h1 = max(rect1.size.height, rect1.size.width);
    auto h2 = max(rect2.size.height, rect2.size.width);
    // auto darea = abs(area1-area2);
    auto aarea = (area1 + area2) / 2;
    auto angle_diff = abs(rect1.angle - rect2.angle);
    if (min(area1, area2) / max(area1, area2) < 0.3)
    {
        flag = false;
    }
    else if (angle_diff > 20)
    {
        flag = false;
    }

    Vec4f line;
    vector<Point2f> points;
    Point2f points1[4];
    Point2f points2[4];
    rect1.points(points1);
    rect2.points(points2);
    for (int i = 0; i < 4; i++)
    {
        points.push_back(points1[i]);
        points.push_back(points2[i]);
    }
    fitLine(points, line, DIST_L2, 0, 0.01, 0.01);
    auto k = line[1] / line[0];
    if (abs(k) > 0.8)
    {
        flag = false;
    }
    // if(aarea > 1500 and aarea*5 < pi/4* distance*distance) flag = false;
    if (distance > 3.5 * max(h1, h2))
        flag = false;
    return flag;
}


/**
 * @brief 筛选装甲板
 *
 * @param 装甲板的四个顶点
 * @param 输入图像（用于显示信息）
 * @return 是与否
 */
bool is_armor(vector<Point2f> points, Mat &input_frame)
{
    bool flag = true;
    RotatedRect r_rect = minAreaRect(points);
    Rect rect = boundingRect(points);
    auto h_w_prop = max(r_rect.size.height, r_rect.size.width) / min(r_rect.size.height, r_rect.size.width);
    auto area = contourArea(points);
    auto r_rect_area = r_rect.size.height * r_rect.size.width;
    // 如果最小矩形宽高比不对
    if (h_w_prop > 2)
    {
        // 青色
        // putText(input_frame, to_string_with_precision(h_w_prop, 2), points[0], FONT_ITALIC, 0.5, Scalar(255, 255, 0));
        flag = false;
    }
    // 如果面积过小，筛除
    else if (area < 10)
    {
        flag = false;
        // 紫色
        // putText(input_frame, to_string_with_precision(contourArea(points), 2), points[2], FONT_ITALIC, 0.5, Scalar(255, 0, 255));
    }
    return flag;
}

/**
 * @brief 根据灯条对找装甲板
 *
 * @param 灯条1
 * @param 灯条2
 * @param 输入图像（用于显示信息）
 * @return 装甲板的四个定点
 */
vector<Point2f> find_armor(RotatedRect rect1, RotatedRect rect2, Mat input_frame)
{
    auto center = (rect1.center + rect2.center) / 2;
    Point2f points1[4];
    rect1.points(points1);
    Point2f points2[4];
    rect2.points(points2);
    vector<point_distance> points, ps1, ps2;
    for (int i = 0; i < 8; i++)
    {
        if (i < 4)
        {
            auto point = points1[i % 4];
            auto distance = calculate_distance(point, center);
            ps1.push_back(point_distance(point, distance));
        }
        else
        {
            auto point = points2[i % 4];
            auto distance = calculate_distance(point, center);
            ps2.push_back(point_distance(point, distance));
        }
    }
    // 分别筛选出两个矩形里离中心点最近的两个点
    sort(ps1.begin(), ps1.end(), [](const point_distance &a, const point_distance &b)
         { return a.distance < b.distance; });
    sort(ps2.begin(), ps2.end(), [](const point_distance &a, const point_distance &b)
         { return a.distance < b.distance; });
    vector<Point2f> armor_points;
    armor_points.push_back(ps1.at(0).point);
    armor_points.push_back(ps1.at(1).point);
    armor_points.push_back(ps2.at(0).point);
    armor_points.push_back(ps2.at(1).point);
    return armor_points;
}

/**
 * @brief 根据灯条对找装甲板
 *
 * @param 图上所有灯条
 * @param 输入图像（用于显示信息）
 * @return 图上所有的装甲板的四个定点
 */
vector<vector<Point2f>> find_armors(vector<RotatedRect> rects, Mat input_frame)
{
    vector<vector<Point2f>> armors;
    if (!rects.empty())
        for (int i = 0; i < (rects.size() - 1); i++)
        {
            RotatedRect rect1 = rects.at(i);
            for (int j = i + 1; j < rects.size(); j++)
            {
                RotatedRect rect2 = rects.at(j);
                // 是否是灯条对
                if (!is_light_pairs(rect1, rect2, input_frame))
                    continue;
                vector<RotatedRect> rect_pair = {rect1, rect2};
                vector<Point2f> armor = find_armor(rect1, rect2, input_frame);
                // 是否是装甲板
                if (!is_armor(armor, input_frame))
                    continue;
                auto center = (rect1.center + rect2.center) / 2;
                sort_points(armor, center);
                armors.push_back(armor);
            }
        }
    return armors;
}

/**
 * @brief 给装甲板四个定点重新按逆时针顺序排序（第三象限开始）
 *
 * @param 装甲板四个顶点
 * @param 装甲板中心
 */
void sort_points(vector<Point2f> &points, Point2f &center)
{
    sort(points.begin(), points.end(), [&](const Point2f &a, const Point2f &b)
         { return compare_angle(center, a, b); });
    for (int i = 0; i < points.size(); i++)
    {
        auto point = points.at(i);
        // cout<<point.x<<" "<<point.y<<endl;
    }
    // cout<<points.size()<<endl;
}


/**
 * @brief PnP求解
 *
 * @param 装甲板四个顶点
 * @param 输入图像（用于显示信息）
 */
void solve_pose_and_mat(vector<Point2f> points, Mat input_frame)
{
    vector<Point3f> truepoints_little = {

        Point3f(-0.0675, -0.03125, 0),
        Point3f(0.0675, -0.03125, 0),
        Point3f(0.0675, 0.03125, 0),
        Point3f(-0.0675, 0.03125, 0)};

    vector<Point3f> truepoints_big = {
        Point3f(-0.115, -0.03125, 0),
        Point3f(0.115, -0.03125, 0),
        Point3f(0.115, 0.03125, 0),
        Point3f(-0.115, 0.03125, 0)};

    vector<double> cameraMatrix = {1.3859739625395162e+03, 0., 9.3622464596653492e+02,
                                   0., 1.3815353250336800e+03, 4.9459467170828475e+02,
                                   0., 0., 1.};
    Mat dc = Mat::zeros(1, 5, CV_64F);
    Mat cm(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            cm.at<double>(i, j) = cameraMatrix[i * 3 + j];
        }
    }
    auto rect = minAreaRect(points);
    auto height = max(rect.size.height, rect.size.width);
    auto width = min(rect.size.height, rect.size.width);
    auto prop = height / width;
    vector<Point3f> truepoints;
    if (prop > 2)
    {
        truepoints = truepoints_little;
    }
    else
    {
        truepoints = truepoints_big;
    }
    Mat r(3, 3, CV_64F);
    Mat t(1, 3, CV_64F);
    solvePnP(truepoints, points, cm, dc, r, t);
    Mat vec(1, 3, CV_64F);
    Mat mat = r.t();
    auto distance = sqrt(pow(t.at<double>(0, 0), 2) + pow(t.at<double>(0, 1), 2) + pow(t.at<double>(0, 2), 2));
    Rodrigues(mat, vec);
    putText(input_frame, "distance=" + to_string(distance), points[0], FONT_ITALIC, 0.5, Scalar(255, 255, 255));
    putText(input_frame, "vec=" + to_string(vec.at<double>(0, 0)) + " " + to_string(vec.at<double>(0, 1)) + " " + to_string(vec.at<double>(0, 2)), points[2], FONT_ITALIC, 0.5, Scalar(255, 255, 255));
}