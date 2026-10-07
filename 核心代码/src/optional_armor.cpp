#include "optional_armor.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>

namespace {

constexpr int kMinLightColorDifference = 45;
constexpr int kMinLightBrightness = 85;
constexpr int kMaxWhiteColorDifference = 45;
constexpr double kMinNumberPixelRatio = 0.025;
constexpr double kMinTemplateCorrelation = 0.80;
constexpr double kMaxReprojectionErrorPx = 2.0;

// 短边中点是灯条的两端；保留斜率方向，避免旋转时把端点顺序弄反。
void findLightEndpoints(const cv::RotatedRect& rectangle,
                        cv::Point2f& top,
                        cv::Point2f& bottom) {
    cv::Point2f corners[4];
    rectangle.points(corners);

    int shortest_edge_index = 0;
    double shortest_edge_length = 1e20;
    for (int index = 0; index < 4; ++index) {
        const double edge_length = cv::norm(corners[(index + 1) % 4] - corners[index]);
        if (edge_length < shortest_edge_length) {
            shortest_edge_length = edge_length;
            shortest_edge_index = index;
        }
    }

    const int edge = shortest_edge_index;
    top = (corners[edge] + corners[(edge + 1) % 4]) * 0.5f;
    bottom = (corners[(edge + 2) % 4] + corners[(edge + 3) % 4]) * 0.5f;
    if (top.y > bottom.y) {
        std::swap(top, bottom);
    }
}

std::vector<Light> findLights(const cv::Mat& image, cv::Mat& mask) {
    // 通道差排除白色号码与灰色背景，亮度条件排除暗处的颜色噪声。
    std::vector<cv::Mat> channels;
    cv::split(image, channels);

    cv::Mat blue_difference;
    cv::Mat red_difference;
    cv::subtract(channels[0], channels[2], blue_difference);
    cv::subtract(channels[2], channels[0], red_difference);

    const cv::Mat blue_mask = (blue_difference > kMinLightColorDifference) &
                              (channels[0] > kMinLightBrightness);
    const cv::Mat red_mask = (red_difference > kMinLightColorDifference) &
                             (channels[2] > kMinLightBrightness);
    mask = blue_mask | red_mask;

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    std::vector<Light> lights;
    for (const auto& contour : contours) {
        if (contour.size() < 6 || cv::contourArea(contour) < 5) {
            continue;
        }

        const cv::RotatedRect rectangle = cv::minAreaRect(contour);
        const float length = std::max(rectangle.size.width, rectangle.size.height);
        const float width = std::min(rectangle.size.width, rectangle.size.height);
        if (length < 6 || width < 0.5 || length / width < 2 || length / width > 30) {
            continue;
        }

        cv::Point2f top;
        cv::Point2f bottom;
        findLightEndpoints(rectangle, top, bottom);
        if (std::abs(bottom.y - top.y) < 0.65 * length) {
            continue;
        }

        const cv::Vec3b center_color = image.at<cv::Vec3b>(cv::Point(rectangle.center));
        const int color = center_color[0] > center_color[2] ? 0 : 1;
        lights.push_back({top, bottom, rectangle.center, length, width, color});
    }

    std::sort(lights.begin(), lights.end(), [](const Light& left, const Light& right) {
        return left.center.x < right.center.x;
    });
    return lights;
}

// 把两条灯之间的区域拉正，检查中间是否存在白色号码，减少跨板误配。
double whiteNumberRatio(const cv::Mat& image, const Light& left, const Light& right) {
    const std::vector<cv::Point2f> source{
        left.top, right.top, right.bottom, left.bottom,
    };
    const std::vector<cv::Point2f> destination{
        {0, 0}, {99, 0}, {99, 59}, {0, 59},
    };

    cv::Mat patch;
    cv::warpPerspective(image, patch,
                        cv::getPerspectiveTransform(source, destination), {100, 60});

    // 只统计中央区域，避开灯条和暗边；白色要求三通道差较小。
    int white_pixels = 0;
    int total_pixels = 0;
    for (int y = 9; y < 51; ++y) {
        for (int x = 27; x < 73; ++x) {
            const cv::Vec3b pixel = patch.at<cv::Vec3b>(y, x);
            const int brightest_channel = std::max({pixel[0], pixel[1], pixel[2]});
            const int darkest_channel = std::min({pixel[0], pixel[1], pixel[2]});
            if (brightest_channel > 100 &&
                brightest_channel - darkest_channel < kMaxWhiteColorDifference) {
                ++white_pixels;
            }
            ++total_pixels;
        }
    }
    return static_cast<double>(white_pixels) / total_pixels;
}

// 灯条外框会受遮挡影响。使用固定号码模板的四角建立可靠的二维/三维对应。
bool recoverNumberCorners(const cv::Mat& image,
                          const Light& left,
                          const Light& right,
                          const Calibration& calibration,
                          std::vector<cv::Point2f>& image_corners,
                          double& correlation) {
    const float average_light_height = (left.length_px + right.length_px) * 0.5f;
    const cv::Point2f pair_center = (left.center + right.center) * 0.5f;
    const float light_spacing = right.center.x - left.center.x;

    // 1. 在灯条附近提取完整号码；上下留白容纳号码高于灯条的部分。
    cv::Rect search_region(
        static_cast<int>(left.center.x - light_spacing * 0.12f),
        static_cast<int>(pair_center.y - average_light_height * 1.5f),
        static_cast<int>(light_spacing * 1.24f),
        static_cast<int>(average_light_height * 3.0f));
    search_region &= cv::Rect(0, 0, image.cols, image.rows);
    if (search_region.width < 5 || search_region.height < 10) {
        return false;
    }

    cv::Mat white_mask(search_region.size(), CV_8U, cv::Scalar(0));
    for (int y = 0; y < search_region.height; ++y) {
        for (int x = 0; x < search_region.width; ++x) {
            const cv::Vec3b pixel = image.at<cv::Vec3b>(search_region.y + y,
                                                       search_region.x + x);
            const int brightest_channel = std::max({pixel[0], pixel[1], pixel[2]});
            const int darkest_channel = std::min({pixel[0], pixel[1], pixel[2]});
            if (darkest_channel > 130 &&
                brightest_channel - darkest_channel < kMaxWhiteColorDifference) {
                white_mask.at<unsigned char>(y, x) = 255;
            }
        }
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(white_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
        return false;
    }

    const auto largest_contour = std::max_element(
        contours.begin(), contours.end(),
        [](const auto& first, const auto& second) {
            return cv::contourArea(first) < cv::contourArea(second);
        });
    const cv::Rect number_bounds = cv::boundingRect(*largest_contour);
    if (number_bounds.height < average_light_height * 1.15 ||
        number_bounds.width < light_spacing * 0.16 ||
        number_bounds.width > light_spacing * 0.85 ||
        cv::contourArea(*largest_contour) < 12) {
        return false;
    }

    const cv::Point2f number_center(
        search_region.x + number_bounds.x + number_bounds.width * 0.5f,
        search_region.y + number_bounds.y + number_bounds.height * 0.5f);
    if (std::abs(number_center.x - pair_center.x) > light_spacing * 0.25f) {
        return false;
    }

    // 2. 先用号码包围盒缩放到模板尺寸，再由 ECC 恢复旋转造成的透视形变。
    const cv::Rect template_bounds = calibration.number_template_bounds;
    const double scale_x = static_cast<double>(template_bounds.width) / number_bounds.width;
    const double scale_y = static_cast<double>(template_bounds.height) / number_bounds.height;
    const cv::Matx33d roi_to_normalized(
        scale_x, 0, template_bounds.x - scale_x * number_bounds.x,
        0, scale_y, template_bounds.y - scale_y * number_bounds.y,
        0, 0, 1);

    cv::Mat normalized_number;
    cv::warpPerspective(white_mask, normalized_number, cv::Mat(roi_to_normalized),
                        calibration.number_template.size());

    cv::Mat reference_float;
    cv::Mat observed_float;
    calibration.number_template.convertTo(reference_float, CV_32F, 1.0 / 255);
    normalized_number.convertTo(observed_float, CV_32F, 1.0 / 255);
    cv::Mat template_to_normalized = cv::Mat::eye(3, 3, CV_32F);
    try {
        correlation = cv::findTransformECC(
            reference_float, observed_float, template_to_normalized, cv::MOTION_HOMOGRAPHY,
            {cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 70, 1e-5}, cv::noArray(), 5);
    } catch (const cv::Exception&) {
        return false;
    }
    if (correlation < kMinTemplateCorrelation) {
        return false;
    }

    // 3. 模板坐标 → 归一化图像 → 搜索区域 → 原图，得到 PnP 使用的图像角点。
    cv::Mat alignment_double;
    template_to_normalized.convertTo(alignment_double, CV_64F);
    const cv::Mat roi_to_image = (cv::Mat_<double>(3, 3) <<
        1, 0, search_region.x,
        0, 1, search_region.y,
        0, 0, 1);
    const cv::Mat template_to_image =
        roi_to_image * cv::Mat(roi_to_normalized).inv() * alignment_double;

    const float template_width = static_cast<float>(calibration.number_template.cols);
    const float template_height = static_cast<float>(calibration.number_template.rows);
    const std::vector<cv::Point2f> template_corners{
        {0, 0}, {template_width, 0},
        {template_width, template_height}, {0, template_height},
    };
    cv::perspectiveTransform(template_corners, image_corners, template_to_image);
    return true;
}

bool estimateArmorPose(const std::vector<cv::Point3f>& model_points,
                       const std::vector<cv::Point2f>& image_points,
                       const Calibration& calibration,
                       Pose& best_pose) {
    std::vector<cv::Mat> rotation_candidates;
    std::vector<cv::Mat> translation_candidates;
    const int solution_count = cv::solvePnPGeneric(
        model_points, image_points, calibration.camera_matrix,
        calibration.distortion_coefficients, rotation_candidates,
        translation_candidates, false, cv::SOLVEPNP_IPPE);

    // 平面 PnP 可能有多个解，取正深度且重投影误差最小的解。
    double best_error = 1e20;
    for (int index = 0; index < solution_count; ++index) {
        cv::Vec3d rotation_vector;
        cv::Vec3d translation_vector;
        rotation_candidates[index].copyTo(rotation_vector);
        translation_candidates[index].copyTo(translation_vector);
        if (translation_vector[2] <= 0) {
            continue;
        }

        cv::solvePnPRefineLM(model_points, image_points, calibration.camera_matrix,
                             calibration.distortion_coefficients,
                             rotation_vector, translation_vector);
        if (translation_vector[2] <= 0 || !cv::checkRange(cv::Mat(translation_vector))) {
            continue;
        }

        std::vector<cv::Point2f> projected_points;
        cv::projectPoints(model_points, rotation_vector, translation_vector,
                          calibration.camera_matrix, calibration.distortion_coefficients,
                          projected_points);
        double reprojection_error = 0.0;
        for (int corner = 0; corner < 4; ++corner) {
            reprojection_error += cv::norm(projected_points[corner] - image_points[corner]);
        }
        reprojection_error /= 4;

        // PnP 原点在号码模板中心，距离评估的参考点在灯条中心。
        cv::Matx33d rotation_matrix;
        cv::Rodrigues(rotation_vector, rotation_matrix);
        const cv::Vec3d target_position =
            translation_vector + rotation_matrix * calibration.target_offset_m;

        if (reprojection_error < best_error) {
            best_error = reprojection_error;
            best_pose.rotation_vector = rotation_vector;
            best_pose.translation_vector = target_position;
            best_pose.distance_m = cv::norm(target_position);
            best_pose.reprojection_error_px = reprojection_error;
        }
    }

    return best_error < kMaxReprojectionErrorPx &&
           best_pose.distance_m > 0.1 && best_pose.distance_m < 30;
}

}  // namespace

Calibration Calibration::load(const std::string& path) {
    const cv::FileStorage storage(path, cv::FileStorage::READ);
    if (!storage.isOpened()) {
        throw std::runtime_error("Cannot read camera calibration");
    }

    Calibration calibration;
    storage["camera_matrix"] >> calibration.camera_matrix;
    storage["distortion"] >> calibration.distortion_coefficients;
    storage["image_width"] >> calibration.image_width;
    storage["image_height"] >> calibration.image_height;
    storage["armor_width_m"] >> calibration.armor_width_m;
    storage["armor_height_m"] >> calibration.armor_height_m;
    storage["pattern_width_m"] >> calibration.pattern_width_m;
    storage["pattern_height_m"] >> calibration.pattern_height_m;

    cv::Mat target_offset;
    storage["target_offset"] >> target_offset;
    target_offset.copyTo(calibration.target_offset_m);

    std::string template_filename;
    storage["pattern_image"] >> template_filename;
    const auto template_path = std::filesystem::path(path).parent_path() / template_filename;
    calibration.number_template = cv::imread(template_path.string(), cv::IMREAD_GRAYSCALE);
    if (calibration.number_template.empty()) {
        throw std::runtime_error("Missing CAD number template");
    }

    cv::threshold(calibration.number_template, calibration.number_template,
                  130, 255, cv::THRESH_BINARY);
    std::vector<cv::Point> template_pixels;
    cv::findNonZero(calibration.number_template, template_pixels);
    calibration.number_template_bounds = cv::boundingRect(template_pixels);

    if (calibration.camera_matrix.rows != 3 || calibration.camera_matrix.cols != 3 ||
        calibration.image_width <= 0 || calibration.image_height <= 0 ||
        calibration.armor_width_m <= 0 || calibration.armor_height_m <= 0 ||
        calibration.pattern_width_m <= 0 || calibration.pattern_height_m <= 0 ||
        calibration.number_template_bounds.width < 8 ||
        calibration.number_template_bounds.height < 8 ||
        !cv::checkRange(calibration.camera_matrix) ||
        !cv::checkRange(calibration.distortion_coefficients)) {
        throw std::runtime_error("Invalid calibration");
    }
    return calibration;
}

std::vector<Pose> detectOptionalArmor(const cv::Mat& image,
                                      const Calibration& calibration,
                                      cv::Mat& mask) {
    const auto lights = findLights(image, mask);
    std::vector<Pose> poses;

    // 号码模板的三维角点，顺序与 recoverNumberCorners 输出的二维角点一致。
    const float half_width = static_cast<float>(calibration.pattern_width_m * 0.5);
    const float half_height = static_cast<float>(calibration.pattern_height_m * 0.5);
    const std::vector<cv::Point3f> model_points{
        {-half_width, -half_height, 0}, {half_width, -half_height, 0},
        {half_width, half_height, 0}, {-half_width, half_height, 0},
    };

    // 只尝试从左到右相邻的灯条，避免跨过中间灯条配出另一块板。
    for (std::size_t index = 0; index + 1 < lights.size(); ++index) {
        const Light& left = lights[index];
        const Light& right = lights[index + 1];
        const double average_height = (left.length_px + right.length_px) * 0.5;
        const double spacing_ratio = (right.center.x - left.center.x) / average_height;
        const double height_ratio = std::min(left.length_px, right.length_px) /
                                    std::max(left.length_px, right.length_px);
        if (left.color != right.color || height_ratio < 0.50 ||
            spacing_ratio < 0.20 || spacing_ratio > 4.0 ||
            std::abs(left.center.y - right.center.y) > average_height * 0.85) {
            continue;
        }

        if (whiteNumberRatio(image, left, right) < kMinNumberPixelRatio) {
            continue;
        }

        std::vector<cv::Point2f> number_corners;
        double correlation = 0.0;
        if (!recoverNumberCorners(image, left, right, calibration, number_corners, correlation)) {
            continue;
        }

        Pose pose;
        if (!estimateArmorPose(model_points, number_corners, calibration, pose)) {
            continue;
        }
        pose.light_corners = {left.top, right.top, right.bottom, left.bottom};
        pose.score = correlation + 0.2 * spacing_ratio;
        poses.push_back(pose);
    }
    return poses;
}

