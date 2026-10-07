#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

enum class LightColor {
    unknown,
    red,
    blue,
};

struct DetectorConfig {
    int binary_threshold = 125;
    int close_width = 3;
    int close_height = 7;
    int open_width = 3;
    int open_height = 3;

    double min_contour_area = 18.0;
    double max_contour_area = 12000.0;
    double min_aspect_ratio = 2.0;
    double max_aspect_ratio = 18.0;
    double min_fill_ratio = 0.35;
    double max_tilt_degrees = 35.0;
    double min_color_delta = 35.0;

    double min_height_ratio = 0.60;
    double max_angle_difference = 18.0;
    double max_center_y_ratio = 0.60;
    double min_center_x_ratio = 0.60;
    double max_center_x_ratio = 6.00;
    int min_number_region_gray = 135;
    double min_number_region_bright_ratio = 0.015;
    int max_armors_per_frame = 1;
};

struct LightBar {
    std::vector<cv::Point> contour;
    cv::RotatedRect rect;
    LightColor color = LightColor::unknown;
    double length = 0.0;
    double width = 0.0;
    double tilt_degrees = 0.0;
    double fill_ratio = 0.0;
};

struct ArmorCandidate {
    cv::RotatedRect rect;
    LightColor color = LightColor::unknown;
    std::size_t left_light = 0;
    std::size_t right_light = 0;
    double number_region_bright_ratio = 0.0;
    double score = 0.0;
};

struct DetectionResult {
    cv::Mat gray;
    cv::Mat binary;
    cv::Mat morph;
    cv::Mat annotated;
    std::vector<LightBar> lights;
    std::vector<ArmorCandidate> armors;
};

class ArmorDetector {
public:
    explicit ArmorDetector(DetectorConfig config = {});

    static DetectorConfig loadConfig(const std::string& path);
    DetectionResult detect(const cv::Mat& frame) const;

private:
    cv::Mat preprocess(const cv::Mat& frame,
                       cv::Mat& gray,
                       cv::Mat& binary) const;
    std::vector<LightBar> findLightBars(const cv::Mat& frame,
                                        const cv::Mat& morph) const;
    std::vector<ArmorCandidate> matchArmors(
        const cv::Mat& frame,
        const std::vector<LightBar>& lights) const;
    void drawDetections(cv::Mat& image,
                        const std::vector<LightBar>& lights,
                        const std::vector<ArmorCandidate>& armors) const;

    DetectorConfig config_;
};

cv::Mat makeDashboard(const cv::Mat& original,
                      const DetectionResult& result,
                      const cv::Size& panel_size = cv::Size(640, 360));

