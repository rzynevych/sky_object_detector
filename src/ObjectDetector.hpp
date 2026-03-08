#ifndef OBJECT_DETECTOR_HPP
#define OBJECT_DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <vector>
#include <set>
#include <string>
#include <cstdint>

class ObjectDetector
{
public:

    struct Box4c {
        int minX, minY, maxX, maxY;
        Box4c() : minX(INT_MAX), minY(INT_MAX), maxX(INT_MIN), maxY(INT_MIN) {}

        Box4c(int _minX, int _minY, int _maxX, int _maxY) : minX(_minX), minY(_minY), maxX(_maxX), maxY(_maxY) {}

        bool operator<(const Box4c &other) const 
        {
            return ((minX << 16) | minY) < ((other.minX << 16) | other.minY); // compare by minX, then by minY
        }

        std::string to_string()
        {
            std::stringstream ss;

            ss << "(" << minX << ", " << minY << ", " << maxX << ", " << maxY << ")";
            return ss.str();
        }
    };

public:
    explicit ObjectDetector(int threshold_high = 200, int treshold_low = 150, int gdist = 100, int smf = 20);

    std::vector<cv::Rect2i> process_frame(cv::Mat& frame);

private:

    // Detection logic
    bool check_distance(const Box4c& b1, const Box4c& b2);
    std::vector<Box4c> group_areas(std::vector<Box4c>& areas);
    std::vector<cv::Rect2i> filter_non_isolated(cv::Mat& lowmask, std::vector<Box4c>& areas);
    std::vector<cv::Rect2i> find_objects(cv::Mat& gradient_img, int threshold_low, int threshold_high);

private:
    // Configuration
    int _threshold_low;
    int _threshold_high;
    int _skyline_margin_factor;
    int _grouping_distance;
    int _rf = 4; // resize factor for flood fill

    // Frame parameters
    int _height;
    int _width;

    // Skyline buffer
    std::vector<int> _skyline;

    // OpenCV buffers
    cv::Mat _gray_img;
    cv::Mat _gradient_img;
    cv::Mat _color_img;
    cv::Mat _fldf_img;
    cv::Mat _skyline_mask;
    cv::Mat _gradient_magnitude;

    cv::Mat _sobelx;
    cv::Mat _sobely;
};

#endif // OBJECT_DETECTOR_HPP