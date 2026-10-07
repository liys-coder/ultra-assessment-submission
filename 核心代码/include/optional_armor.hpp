#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <string>
#include <vector>

// 相机内参与 CAD 号码模板；所有物理尺寸均使用米。
struct Calibration {
    cv::Mat camera_matrix;
    cv::Mat distortion_coefficients;
    cv::Mat number_template;
    cv::Rect number_template_bounds;

    // 从号码模板原点指向灯条中心的固定偏移，在模板坐标系中表示。
    cv::Vec3d target_offset_m{};
    int image_width = 0;
    int image_height = 0;
    double armor_width_m = 0.0;
    double armor_height_m = 0.0;
    double pattern_width_m = 0.0;
    double pattern_height_m = 0.0;

    static Calibration load(const std::string& path);
};

struct Light {
    cv::Point2f top;
    cv::Point2f bottom;
    cv::Point2f center;
    float length_px = 0.0f;
    float width_px = 0.0f;
    int color = 0;  // 0：蓝色，1：红色。
};

struct Pose {
    // 顺序：左上、右上、右下、左下。用于绘图与评估，PnP 使用号码模板角点。
    std::array<cv::Point2f, 4> light_corners;
    cv::Vec3d rotation_vector{};
    cv::Vec3d translation_vector{};  // 灯条中心在相机坐标系中的位置，单位：米。
    double distance_m = 0.0;
    double reprojection_error_px = 0.0;
    double score = 0.0;
};

// 只使用图像和固定标定数据。返回所有通过筛选的候选，mask 为红蓝灯条掩码。
std::vector<Pose> detectOptionalArmor(const cv::Mat& image,
                                      const Calibration& calibration,
                                      cv::Mat& mask);
