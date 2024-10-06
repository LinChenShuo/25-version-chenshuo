#include <iostream>
#include <opencv2/opencv.hpp>
#include "mylib.h"

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    // 源文件
    const string input_filename = "./zimiao_test.mp4";
    // 输出文件
    const string output_filename = "./zimiao_coutours.mp4";
    // 红色提取
    const string r_filename = "./r.mp4";
    // 蓝色提取
    const string b_filename = "./b.mp4";
    VideoCapture capture(input_filename);
    VideoWriter writer(output_filename,
                       capture.get(CAP_PROP_FOURCC),
                       capture.get(CAP_PROP_FPS),
                       Size(capture.get(CAP_PROP_FRAME_WIDTH), capture.get(CAP_PROP_FRAME_HEIGHT)),
                       true);
    VideoWriter writerb(b_filename,
                        capture.get(CAP_PROP_FOURCC),
                        capture.get(CAP_PROP_FPS),
                        Size(capture.get(CAP_PROP_FRAME_WIDTH), capture.get(CAP_PROP_FRAME_HEIGHT)),
                        false);
    VideoWriter writerr(r_filename,
                        capture.get(CAP_PROP_FOURCC),
                        capture.get(CAP_PROP_FPS),
                        Size(capture.get(CAP_PROP_FRAME_WIDTH), capture.get(CAP_PROP_FRAME_HEIGHT)),
                        false);
    Mat input_frame;
    if (!capture.isOpened())
    {
        cout << "打开视频失败！" << endl;
        return -1;
    }
    while (true)
    {
        capture.read(input_frame);
        if (input_frame.empty())
            break;
        Mat r_mask;
        Mat b_mask;
        // 输出提取
        split_img_to_masks(input_frame, r_mask, b_mask);
        // 找灯条
        vector<RotatedRect> r_rects = find_rect_lights(input_frame, r_mask, false);
        vector<RotatedRect> b_rects = find_rect_lights(input_frame, b_mask, false);
        // 找装甲板
        auto r_armors = find_armors(r_rects, input_frame);
        auto b_armors = find_armors(b_rects, input_frame);
        // 画装甲板并求解位姿
        for (int i = 0; i < r_armors.size(); i++)
        {
            auto armor = r_armors.at(i);
            for (int j = 0; j < armor.size(); j++)
            {
                circle(input_frame, armor.at(j), 2, Scalar(0, 255, 0), 2);
                line(input_frame, armor[j], armor[(j + 1) % 4], Scalar(0, 255, 0), 2);
            }
            solve_pose_and_mat(armor, input_frame);
        }
        for (int i = 0; i < b_armors.size(); i++)
        {
            auto armor = b_armors.at(i);
            for (int j = 0; j < armor.size(); j++)
            {
                circle(input_frame, armor.at(j), 2, Scalar(0, 255, 0), 2);
                line(input_frame, armor[j], armor[(j + 1) % 4], Scalar(0, 255, 0), 2);
            }
            solve_pose_and_mat(armor, input_frame);
        }

        imshow("contours", input_frame);
        if (waitKey(int(1000 / capture.get(CAP_PROP_FPS))) == 'q')
            break;
        writer << input_frame;
        writerb << b_mask;
        writerr << r_mask;
    }
    capture.release();
    writer.release();
    writerb.release();
    writerr.release();
    cout << "播放结束" << endl;
    return 0;
}
