# Object Detection Using Gradient & Skyline Filtering

[![Demo](media/demo.gif)](https://www.youtube.com/watch?v=FDf00D14dVw)

## Overview

This project implements a lightweight real-time object detection algorithm focused on detecting contrast objects above a dynamically computed skyline.

Unlike ML-based approaches, this method is fully classical and deterministic. It relies on gradient magnitude analysis and region filtering techniques to detect small distant objects such as:

* UAVs
* Birds
* Aircraft
* Other high-contrast objects in the sky

The system is optimized for performance and designed to integrate into real-time video processing pipelines.

## Key Features

* Real-time contrast object detection in sky regions
* Produces bounding boxes for detected objects
* No machine learning dependencies
* Fully deterministic behavior

## Algorithm Pipeline

### 1. Grayscale Conversion

Input frames are converted from BGR to grayscale to reduce computational cost and simplify gradient computation.

### 2. Gradient Computation

* Sobel operator is applied in X and Y directions
* Gradient magnitude is computed using `cv::magnitude`
* High-threshold binary mask is generated to isolate strong contrast regions

This step highlights sharp intensity transitions corresponding to object boundaries.

### 3. Connected Component Extraction

Binary mask is processed using:

```cpp
cv::connectedComponentsWithStats(...)
```

For each connected component:

* Bounding box
* Area
* Basic statistics

are extracted efficiently without per-pixel scanning.

### 4. Skyline Detection

A skyline is estimated using a flood-fill–based approach to separate sky and ground regions.

Each detected object is validated using:

```
object.maxY < skyline[x_center] - margin
```

Objects below the skyline are discarded.

This significantly reduces false positives from:

* Trees
* Buildings
* Poles
* Terrain structures

### 5. Region Grouping

Spatially close bounding boxes are merged if their center distance is below a configurable threshold.

This helps combine fragmented detections of the same object.

### 6. Isolation Filtering

Border analysis is performed using a lower gradient threshold mask.

Objects are rejected if they are connected to surrounding background structures.

This reduces false positives caused by:

* High-contrast vertical structures
* Edge fragments from large objects

## Possible Improvements

Several extensions could further improve robustness and detection quality:

* **Temporal filtering** to enforce detection consistency across frames.
* **Motion-based filtering** to suppress static background structures.
* **Adaptive gradient thresholds** to better handle varying lighting conditions.
* **Morphological preprocessing** to reduce noise before connected component extraction.
* **More efficient region grouping** using spatial indexing structures.
* **Multi-scale detection** to improve sensitivity to small or distant objects.

## Requirements

* C++17 or newer
* OpenCV 4.x

## Example Usage

* Build and run the project:
```bash
cmake -B build
cmake --build build
./build/SkyObjectDetector <video-file-name>
```
* Or include ObjectDetector class in your project:

```cpp
#include "ObjectDetector.hpp"

int main(int argc, char **argv) 
{
    cv::VideoCapture cap("video.mp4");

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the video file!" << std::endl;
        return 1;
    }

    ObjectDetector detector(
        220, // threshold_high
        80, // threshold_low
        100, // grouping_distance
        40 // skyline_margin_factor
    );

    cv::Mat frame;
    while (true) {
        cap >> frame;

        if (frame.empty())
            break;

        std::vector<cv::Rect2i> areas = detector.process_frame(frame);

        for (auto &rect : areas) {
            cv::rectangle(frame, rect, cv::Scalar(0, 0, 255), 2);
        }

        cv::imshow("Detection", frame);

        if (cv::waitKey(1) == 'q')
            break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
```

## Parameters

| Parameter           | Description                                                 |
| ------------------- | ----------------------------------------------------------- |
| `threshold_high`    | Threshold for strong gradient mask                          |
| `threshold_low`     | Threshold for isolation filtering                           |
| `gdist`             | Maximum distance for merging regions                        |
| `smf`               | Skyline margin factor (skyline_margin = _height / smf)      |


## Performance Notes

* Uses `CV_32F` Sobel for precision
* Uses `convertScaleAbs` for fast normalization
* Avoids manual bounding box scans (relies on OpenCV stats)
* Designed for optimized builds (`-O2` / `/O2`)

### Typical Performance

* ~10–12 ms per 1080p frame
* Tested on Intel Core i5-12420HX
* Suitable for real-time 60 FPS pipelines depending on I/O overhead
