#include "ObjectDetector.hpp"
#include "utils.h"

#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>

ObjectDetector::ObjectDetector(int threshold_high, int treshold_low, int gdist, int smf)
    : _threshold_high(threshold_high), _threshold_low(treshold_low),
        _skyline_margin_factor(smf), _grouping_distance(gdist)
{}

bool ObjectDetector::check_distance(const Box4c& b1, const Box4c& b2)
{
    int xc1 = (b1.minX + b1.maxX) / 2;
    int yc1 = (b1.minY + b1.maxY) / 2;
    int xc2 = (b2.minX + b2.maxX) / 2;
    int yc2 = (b2.minY + b2.maxY) / 2;

    int dx = xc2 - xc1;
    int dy = yc2 - yc1;

    int sqrdist = dx * dx + dy * dy;

    // std::cout << "check_distance: " << b1.to_string() << ", " << b2.to_string() << ": " << std::sqrt(sqrdist) << std::endl;

    return sqrdist < _grouping_distance * _grouping_distance;
}

std::vector<ObjectDetector::Box4c> ObjectDetector::group_areas(std::vector<Box4c> &areas) 
{
    int window_size = _grouping_distance * 2;
    std::vector<std::vector<int>> groups;
    std::set<int> used;
    for (int i = 0; i < areas.size(); i++) {
        if (used.find(i) != used.end()) {
            continue;
        }
        // std::cout << "i: " << i << " | ";
        std::vector<int> group;
        group.push_back(i);
        int j = i + 1;
        while(j < areas.size() && areas[j].minX - areas[i].minX < window_size) {
            if (used.find(j) != used.end()) {
                j += 1;
                continue;
            }
            for (int k = 0; k < group.size(); k++) {
                // std::cout  << j << " | " << k << " | ";
                if (j != k && check_distance(areas[group[k]], areas[j])) {
                    group.push_back(j);
                    used.insert(j);
                    // std::cout << "inserted | ";
                    break;
                }
            }
            j++;
        }
        // std::cout << "Group: ";
        // print_vector(group);
        groups.push_back(std::move(group));
    }

    // std::cout << "Areas size:" << areas.size() << std::endl;
    
    std::vector<Box4c> merged_areas;
    for (std::vector<int> &group : groups) {
        auto it = std::min_element(group.begin(), group.end(), [&areas](int a, int b) { return areas[a].minX < areas[b].minX; });
        int minX = areas[*it].minX;
        it = std::max_element(group.begin(), group.end(), [&areas](int a, int b) { return areas[a].maxX < areas[b].maxX; });
        int maxX = areas[*it].maxX;
        it = std::min_element(group.begin(), group.end(), [&areas](int a, int b) { return areas[a].minY < areas[b].minY; });
        int minY = areas[*it].minY;
        it = std::max_element(group.begin(), group.end(), [&areas](int a, int b) { return areas[a].maxY < areas[b].maxY; });
        int maxY = areas[*it].maxY;
        merged_areas.push_back(Box4c(minX, minY, maxX, maxY));
    }

    return merged_areas;
}

std::vector<cv::Rect2i> ObjectDetector::filter_non_isolated(cv::Mat &lowmask, std::vector<Box4c> &areas)
{
    int width = lowmask.cols;
    int height = lowmask.rows;
    
    // std::cout << "areas: ";
    std::vector<cv::Rect2i> filtered_areas;
    for (auto &area : areas) {

        int indent =  (area.maxX - area.minX + area.maxY - area.minY) / 10;
        indent = std::max(width / 120, indent);
        int bw = 5; // border width
        // std::cout << "indent: " << indent << std::endl;
        if (area.minX - indent < 0 || area.minY - indent < 0 || area.maxX + indent >= width || area.maxY + indent >= height) {
            continue;
        }
        int bminX, bmaxX, bminY, bmaxY;
        bminX = area.minX - indent;
        bmaxX = area.maxX + indent;
        bminY = area.minY - indent;
        bmaxY = area.maxY + indent;

        cv::Scalar cv_check_sum = cv::sum(lowmask(cv::Rect2i(bminX, bminY, bmaxX - bminX, bw))).val[0]
                + cv::sum(lowmask(cv::Rect2i(bminX, bmaxY - bw, bmaxX - bminX, bw))).val[0]
                + cv::sum(lowmask(cv::Rect2i(bminX, bminY + bw, bw, bmaxY - bminY - 2*bw))).val[0]
                + cv::sum(lowmask(cv::Rect2i(bmaxX - bw, bminY + bw, bw, bmaxY - bminY - 2*bw))).val[0];

        int check_sum = (int) cv_check_sum[0];
        // std::cout << area.to_string() << " " << check_sum << ", ";
        int treshold = 1 + ((bmaxX - bminX) * (bmaxY - bminY) - (bmaxX - bminX - 2 * bw) * (bmaxY - bminY - 2 * bw)) / 200;
        // std::cout << "treshold: " << treshold << std::endl;
        if (check_sum < treshold) {
            filtered_areas.emplace_back(area.minX, area.minY, area.maxX - area.minX, area.maxY - area.minY);
        }
    }
    // std::cout << std::endl;

    return filtered_areas;
}

std::vector<cv::Rect2i> ObjectDetector::find_objects(
                cv::Mat &gradient_img, int threshold_low, int threshold_high)
{
    static cv::Mat lowmask, highmask;

    // Step 1: Threshold to get high-value mask
    cv::threshold(gradient_img, lowmask, threshold_low, 1, cv::THRESH_BINARY);
    cv::threshold(gradient_img, highmask, threshold_high, 1, cv::THRESH_BINARY);

    // Step 2: Label connected components
    cv::Mat labels, stats, centroids;

    int numComponents = cv::connectedComponentsWithStats(
        highmask,          // binary image
        labels,            // output labels
        stats,             // output statistics
        centroids,         // output centroids
        8,                 // connectivity (4 or 8)
        CV_32S             // label type
    );

    // Step 3: Find bounding boxes for each component
    std::vector<Box4c> boxes(numComponents - 1); // index 0 is background
    for (int i = 1; i < numComponents; ++i) {
        auto& box = boxes[i - 1];
        box.minX = stats.at<int>(i, cv::CC_STAT_LEFT);
        box.minY = stats.at<int>(i, cv::CC_STAT_TOP);
        box.maxX = box.minX + stats.at<int>(i, cv::CC_STAT_WIDTH);
        box.maxY = box.minY + stats.at<int>(i, cv::CC_STAT_HEIGHT);
    }

    std::vector<Box4c> filtered_boxes;
    int skyline_margin = _height / _skyline_margin_factor;
    std::copy_if(boxes.begin(), boxes.end(), std::back_inserter(filtered_boxes),
                [=](Box4c b) { return b.maxY < _rf * _skyline[(b.minX + b.maxX) / (2 * _rf)] - skyline_margin; });

    std::sort(filtered_boxes.begin(), filtered_boxes.end());

    std::vector<Box4c> areas = group_areas(filtered_boxes);
    return filter_non_isolated(lowmask, areas);
}
    
std::vector<cv::Rect2i> ObjectDetector::process_frame(cv::Mat &frame)
{
    get_elapsed_time();

    cv::cvtColor(frame, _gray_img, cv::COLOR_BGR2GRAY);

    _height = _gray_img.rows;
    _width  = _gray_img.cols;

    // print_elapsed_time("sobel start");

    // Apply Sobel operator
    cv::Sobel(_gray_img, _sobelx, CV_32F, 1, 0, 3);
    cv::Sobel(_gray_img, _sobely, CV_32F, 0, 1, 3);
    cv::magnitude(_sobelx, _sobely, _gradient_magnitude);
    
    // Convert to uint8 and img_color

    // u_gradient_magnitude.copyTo(gradient_magnitude);
    // print_elapsed_time("sobel end");
    cv::convertScaleAbs(_gradient_magnitude, _gradient_img);
    cv::cvtColor(_gradient_img, _color_img, cv::COLOR_GRAY2BGR);
    
    cv::Point2i sp1(_width / (_rf * 2), 0);
    cv::Scalar fill_color(0, 255, 0); // Green
    cv::Scalar loDiffv(1, 1, 1);
    cv::Scalar upDiffv(10, 10, 10);

    cv::resize(_color_img, _fldf_img, cv::Size2i( _width/_rf , _height/_rf));
    cv::rectangle(_fldf_img, cv::Rect2i(0, 0, _width / 4, 1), cv::Scalar(0, 0, 0), 2); // draw a top plain line to make filling steady
    cv::floodFill(_fldf_img, sp1, fill_color, 0, loDiffv, upDiffv, 4 | cv::FLOODFILL_FIXED_RANGE);

    // cv::imshow("fill", small_img);

    cv::inRange(_fldf_img, fill_color, fill_color, _skyline_mask);
    _skyline_mask /= 255;

    // print_elapsed_time("before skyline");

    _skyline.resize(_width / _rf, 0);
    std::fill(_skyline.begin(), _skyline.end(), 0);
    for (int y = 0; y < _skyline_mask.rows; ++y) {    
        for (int x = 0; x < _skyline_mask.cols; ++x) {
            if (_skyline_mask.at<uchar>(y, x) == 1) {
                _skyline[x] = y;
            }
        }
    }

    // print_elapsed_time("before search");

    std::vector<cv::Rect2i> areas = find_objects(_gradient_img, _threshold_low, _threshold_high);

    // cv::Mat &output_frame = frame;
    // for (int i = 1; i < _width / _rf; i++) {
    //     cv::line(output_frame, cv::Point2i((i - 1) * _rf, _skyline[i - 1] * _rf), cv::Point2i(i * _rf, _skyline[i] * _rf), cv::Scalar(255, 0, 0), 2);
    // }

    print_elapsed_time("process_frame end");
    return areas;
}