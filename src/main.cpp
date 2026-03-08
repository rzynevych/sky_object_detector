#include "ObjectDetector.hpp"
#include "utils.h"

#include <iostream>

int main(int argc, char **argv) 
{
    static double fps = 0.0;

    // if (argc < 2) {
    //     std::cout << "Usage: " << argv[0] << " <video_file_name>" << std::endl; 
    //     return 0;
    // }

    std::cout << "Start" << std::endl;
    
    // char *filename = argv[1];

    // Open the video file
    // cv::VideoCapture cap("rtspsrc location=rtsp://192.168.1.65:8554/my_video latency=0 ! rtph264depay ! h264parse ! mppvideodec format=BGR ! appsink drop=True sync=False", cv::CAP_GSTREAMER); // replace with your video path
    cv::VideoCapture cap("D:\\Libraries\\Videos\\output.mp4"); // replace with your video path

    // Check if the video was successfully opened
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the video file!" << std::endl;
        return 1;
    }

    cv::Mat frame;

    ObjectDetector detector;
    double start;
    while (true) {
        start = current_time();
        cap >> frame; // read the next frame

        if (frame.empty()) {
            break; // end of video
        }

        std::vector<cv::Rect2i> areas = detector.process_frame(frame);
        for (auto &rect : areas) {
            cv::rectangle(frame, rect, cv::Scalar(0, 0, 255), 2);
        }

        cv::Mat output_frame;
        cv::resize(frame, output_frame, cv::Size2i(1360, 765));

        double frame_time = current_time() - start;
        double current_fps = 1.0 / frame_time;
        double alpha = 0.02; // Smoothing factor for the moving average
        fps = alpha * current_fps + (1 - alpha) * fps; // Exponential moving average for FPS

        std::string fpstext{"FPS: "};
        fpstext += std::to_string((int)fps);
        cv::putText(output_frame, fpstext, cv::Point2i(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1);

        cv::imshow("Video", output_frame);

        // Exit when 'q' is pressed or wait 30ms between frames
        char key = cv::waitKey(1);
        if (key == 'q') {
            break;
        } else if (key == 'p') {
            cv::waitKey();
        }
    }

    cap.release();
    cv::destroyAllWindows();
    std::cout << "Finish" << std::endl;

    return 0;
}