#include "armor_detector.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace {

template <typename T>
void readIfPresent(const cv::FileStorage& storage,
                   const char* key,
                   T& value) {
    const cv::FileNode node = storage[key];
    if (!node.empty()) {
        node >> value;
    }
}

int positiveOdd(int value) {
    value = std::max(value, 1);
    return value % 2 == 0 ? value + 1 : value;
}

double verticalTiltDegrees(const cv::RotatedRect& rect) {
    cv::Point2f points[4];
    rect.points(points);

    cv::Point2f longest_edge;
    double longest_length = -1.0;
    for (int index = 0; index < 4; ++index) {
        const cv::Point2f edge = points[(index + 1) % 4] - points[index];
        const double length = cv::norm(edge);
        if (length > longest_length) {
            longest_length = length;
            longest_edge = edge;
        }
    }

    return std::atan2(std::abs(longest_edge.x),
                      std::abs(longest_edge.y)) * 180.0 / CV_PI;
}

cv::Scalar colorFor(LightColor color) {
    if (color == LightColor::blue) {
        return cv::Scalar(255, 180, 0);
    }
    if (color == LightColor::red) {
        return cv::Scalar(0, 80, 255);
    }
    return cv::Scalar(0, 255, 255);
}

void drawRotatedRect(cv::Mat& image,
                     const cv::RotatedRect& rect,
                     const cv::Scalar& color,
                     int thickness) {
    cv::Point2f points[4];
    rect.points(points);
    for (int index = 0; index < 4; ++index) {
        cv::line(image,
                 points[index],
                 points[(index + 1) % 4],
                 color,
                 thickness,
                 cv::LINE_AA);
    }
}

cv::Mat makePanel(const cv::Mat& source,
                  const std::string& label,
                  const cv::Size& panel_size) {
    cv::Mat color;
    if (source.channels() == 1) {
        cv::cvtColor(source, color, cv::COLOR_GRAY2BGR);
    } else {
        color = source.clone();
    }

    const double scale = std::min(
        static_cast<double>(panel_size.width) / color.cols,
        static_cast<double>(panel_size.height) / color.rows);
    const cv::Size resized_size(
        std::max(1, static_cast<int>(std::round(color.cols * scale))),
        std::max(1, static_cast<int>(std::round(color.rows * scale))));

    cv::Mat resized;
    cv::resize(color, resized, resized_size, 0.0, 0.0, cv::INTER_AREA);

    cv::Mat panel(panel_size, CV_8UC3, cv::Scalar(24, 24, 24));
    const int x = (panel_size.width - resized.cols) / 2;
    const int y = (panel_size.height - resized.rows) / 2;
    resized.copyTo(panel(cv::Rect(x, y, resized.cols, resized.rows)));

    cv::rectangle(panel, cv::Rect(0, 0, panel.cols, 38),
                  cv::Scalar(16, 16, 16), cv::FILLED);
    cv::putText(panel, label, cv::Point(14, 27),
                cv::FONT_HERSHEY_SIMPLEX, 0.72,
                cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    return panel;
}

}  // namespace

ArmorDetector::ArmorDetector(DetectorConfig config)
    : config_(std::move(config)) {}

DetectorConfig ArmorDetector::loadConfig(const std::string& path) {
    cv::FileStorage storage(path, cv::FileStorage::READ);
    if (!storage.isOpened()) {
        throw std::runtime_error("Cannot open detector config: " + path);
    }

    DetectorConfig config;
    readIfPresent(storage, "binary_threshold", config.binary_threshold);
    readIfPresent(storage, "close_width", config.close_width);
    readIfPresent(storage, "close_height", config.close_height);
    readIfPresent(storage, "open_width", config.open_width);
    readIfPresent(storage, "open_height", config.open_height);
    readIfPresent(storage, "min_contour_area", config.min_contour_area);
    readIfPresent(storage, "max_contour_area", config.max_contour_area);
    readIfPresent(storage, "min_aspect_ratio", config.min_aspect_ratio);
    readIfPresent(storage, "max_aspect_ratio", config.max_aspect_ratio);
    readIfPresent(storage, "min_fill_ratio", config.min_fill_ratio);
    readIfPresent(storage, "max_tilt_degrees", config.max_tilt_degrees);
    readIfPresent(storage, "min_color_delta", config.min_color_delta);
    readIfPresent(storage, "min_height_ratio", config.min_height_ratio);
    readIfPresent(storage, "max_angle_difference", config.max_angle_difference);
    readIfPresent(storage, "max_center_y_ratio", config.max_center_y_ratio);
    readIfPresent(storage, "min_center_x_ratio", config.min_center_x_ratio);
    readIfPresent(storage, "max_center_x_ratio", config.max_center_x_ratio);
    readIfPresent(storage, "min_number_region_gray",
                  config.min_number_region_gray);
    readIfPresent(storage, "min_number_region_bright_ratio",
                  config.min_number_region_bright_ratio);
    readIfPresent(storage, "max_armors_per_frame",
                  config.max_armors_per_frame);

    config.binary_threshold = std::clamp(config.binary_threshold, 0, 255);
    config.close_width = positiveOdd(config.close_width);
    config.close_height = positiveOdd(config.close_height);
    config.open_width = positiveOdd(config.open_width);
    config.open_height = positiveOdd(config.open_height);
    config.min_number_region_gray =
        std::clamp(config.min_number_region_gray, 0, 255);
    config.max_armors_per_frame = std::max(config.max_armors_per_frame, 1);
    return config;
}

cv::Mat ArmorDetector::preprocess(const cv::Mat& frame,
                                  cv::Mat& gray,
                                  cv::Mat& binary) const {
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, binary, config_.binary_threshold, 255,
                  cv::THRESH_BINARY);

    cv::Mat morph;
    const cv::Mat close_kernel = cv::getStructuringElement(
        cv::MORPH_RECT,
        cv::Size(config_.close_width, config_.close_height));
    const cv::Mat open_kernel = cv::getStructuringElement(
        cv::MORPH_RECT,
        cv::Size(config_.open_width, config_.open_height));
    cv::morphologyEx(binary, morph, cv::MORPH_CLOSE, close_kernel);
    cv::morphologyEx(morph, morph, cv::MORPH_OPEN, open_kernel);
    return morph;
}

std::vector<LightBar> ArmorDetector::findLightBars(
    const cv::Mat& frame,
    const cv::Mat& morph) const {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(morph.clone(), contours,
                     cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<LightBar> lights;
    lights.reserve(contours.size());

    for (const auto& contour : contours) {
        const double area = cv::contourArea(contour);
        if (area < config_.min_contour_area ||
            area > config_.max_contour_area || contour.size() < 4) {
            continue;
        }

        const cv::RotatedRect rect = cv::minAreaRect(contour);
        const double length = std::max(rect.size.width, rect.size.height);
        const double width = std::min(rect.size.width, rect.size.height);
        if (width < 1.0) {
            continue;
        }

        const double aspect_ratio = length / width;
        const double fill_ratio = area / (length * width);
        const double tilt = verticalTiltDegrees(rect);
        if (aspect_ratio < config_.min_aspect_ratio ||
            aspect_ratio > config_.max_aspect_ratio ||
            fill_ratio < config_.min_fill_ratio ||
            tilt > config_.max_tilt_degrees) {
            continue;
        }

        cv::Mat contour_mask(morph.size(), CV_8UC1, cv::Scalar(0));
        std::vector<std::vector<cv::Point>> one_contour{contour};
        cv::drawContours(contour_mask, one_contour, 0,
                         cv::Scalar(255), cv::FILLED);
        const cv::Scalar mean_bgr = cv::mean(frame, contour_mask);
        const double blue_delta = mean_bgr[0] - mean_bgr[2];
        const double red_delta = mean_bgr[2] - mean_bgr[0];

        LightColor color = LightColor::unknown;
        if (blue_delta >= config_.min_color_delta) {
            color = LightColor::blue;
        } else if (red_delta >= config_.min_color_delta) {
            color = LightColor::red;
        } else {
            continue;
        }

        lights.push_back(LightBar{
            contour, rect, color, length, width, tilt, fill_ratio});
    }

    std::sort(lights.begin(), lights.end(),
              [](const LightBar& left, const LightBar& right) {
                  return left.rect.center.x < right.rect.center.x;
              });
    return lights;
}

std::vector<ArmorCandidate> ArmorDetector::matchArmors(
    const cv::Mat& frame,
    const std::vector<LightBar>& lights) const {
    std::vector<ArmorCandidate> candidates;

    for (std::size_t left_index = 0; left_index < lights.size(); ++left_index) {
        // 只匹配相邻灯条，避免跨过中间灯条产生组合爆炸和大范围误框。
        const std::size_t right_index = left_index + 1;
        if (right_index < lights.size()) {
            const LightBar& left = lights[left_index];
            const LightBar& right = lights[right_index];
            if (left.color != right.color) {
                continue;
            }

            const double average_height = (left.length + right.length) * 0.5;
            const double height_ratio =
                std::min(left.length, right.length) /
                std::max(left.length, right.length);
            const double angle_difference =
                std::abs(left.tilt_degrees - right.tilt_degrees);
            const double center_y_ratio =
                std::abs(left.rect.center.y - right.rect.center.y) /
                average_height;
            const double center_x_ratio =
                std::abs(left.rect.center.x - right.rect.center.x) /
                average_height;

            if (height_ratio < config_.min_height_ratio ||
                angle_difference > config_.max_angle_difference ||
                center_y_ratio > config_.max_center_y_ratio ||
                center_x_ratio < config_.min_center_x_ratio ||
                center_x_ratio > config_.max_center_x_ratio) {
                continue;
            }

            const int roi_left = static_cast<int>(std::round(
                left.rect.center.x + left.width));
            const int roi_right = static_cast<int>(std::round(
                right.rect.center.x - right.width));
            const double center_y =
                (left.rect.center.y + right.rect.center.y) * 0.5;
            const int roi_top = static_cast<int>(std::round(
                center_y - average_height * 0.55));
            const int roi_bottom = static_cast<int>(std::round(
                center_y + average_height * 0.55));

            cv::Rect number_roi(
                cv::Point(std::min(roi_left, roi_right), roi_top),
                cv::Point(std::max(roi_left, roi_right), roi_bottom));
            number_roi &= cv::Rect(0, 0, frame.cols, frame.rows);
            if (number_roi.width < 2 || number_roi.height < 2) {
                continue;
            }

            const cv::Mat region = frame(number_roi);
            std::size_t bright_neutral_pixels = 0;
            for (int row = 0; row < region.rows; ++row) {
                const cv::Vec3b* pixels = region.ptr<cv::Vec3b>(row);
                for (int column = 0; column < region.cols; ++column) {
                    const cv::Vec3b pixel = pixels[column];
                    const int blue = pixel[0];
                    const int green = pixel[1];
                    const int red = pixel[2];
                    const int maximum = std::max({blue, green, red});
                    const int minimum = std::min({blue, green, red});
                    const int gray = static_cast<int>(std::round(
                        0.114 * blue + 0.587 * green + 0.299 * red));
                    if (gray >= config_.min_number_region_gray &&
                        maximum - minimum <= 25) {
                        ++bright_neutral_pixels;
                    }
                }
            }

            const double bright_ratio =
                static_cast<double>(bright_neutral_pixels) /
                static_cast<double>(number_roi.area());
            if (bright_ratio < config_.min_number_region_bright_ratio) {
                continue;
            }

            std::vector<cv::Point2f> points;
            points.reserve(8);
            cv::Point2f corners[4];
            left.rect.points(corners);
            points.insert(points.end(), std::begin(corners), std::end(corners));
            right.rect.points(corners);
            points.insert(points.end(), std::begin(corners), std::end(corners));

            const double score =
                bright_ratio * 4.0 + height_ratio -
                0.20 * center_y_ratio -
                0.02 * angle_difference;
            candidates.push_back(ArmorCandidate{
                cv::minAreaRect(points), left.color, left_index, right_index,
                bright_ratio, score});
        }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const ArmorCandidate& left,
                 const ArmorCandidate& right) {
                  return left.score > right.score;
              });
    if (candidates.size() >
        static_cast<std::size_t>(config_.max_armors_per_frame)) {
        candidates.resize(
            static_cast<std::size_t>(config_.max_armors_per_frame));
    }
    return candidates;
}

void ArmorDetector::drawDetections(
    cv::Mat& image,
    const std::vector<LightBar>& lights,
    const std::vector<ArmorCandidate>& armors) const {
    for (const LightBar& light : lights) {
        std::vector<std::vector<cv::Point>> contours{light.contour};
        cv::drawContours(image, contours, 0, colorFor(light.color),
                         2, cv::LINE_AA);
        drawRotatedRect(image, light.rect, colorFor(light.color), 2);
    }

    for (const ArmorCandidate& armor : armors) {
        drawRotatedRect(image, armor.rect, cv::Scalar(40, 255, 40), 3);
        cv::circle(image, armor.rect.center, 4,
                   cv::Scalar(0, 255, 255), cv::FILLED, cv::LINE_AA);
    }

    const std::string summary =
        "lights: " + std::to_string(lights.size()) +
        "  armors: " + std::to_string(armors.size());
    const int summary_y = std::max(10, image.rows - 52);
    cv::rectangle(image, cv::Rect(10, summary_y, 310, 38),
                  cv::Scalar(16, 16, 16), cv::FILLED);
    cv::putText(image, summary, cv::Point(20, summary_y + 27),
                cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
}

DetectionResult ArmorDetector::detect(const cv::Mat& frame) const {
    if (frame.empty()) {
        throw std::invalid_argument("Cannot detect an empty frame");
    }

    DetectionResult result;
    result.morph = preprocess(frame, result.gray, result.binary);
    result.lights = findLightBars(frame, result.morph);
    result.armors = matchArmors(frame, result.lights);
    result.annotated = frame.clone();
    drawDetections(result.annotated, result.lights, result.armors);
    return result;
}

cv::Mat makeDashboard(const cv::Mat& original,
                      const DetectionResult& result,
                      const cv::Size& panel_size) {
    cv::Mat original_panel = makePanel(original, "1. Original", panel_size);
    cv::Mat gray_panel = makePanel(result.gray, "2. Gray", panel_size);
    cv::Mat morph_panel = makePanel(
        result.morph, "3. Binary + morphology", panel_size);
    cv::Mat detection_panel = makePanel(
        result.annotated, "4. Light bars + armor", panel_size);

    cv::Mat top;
    cv::Mat bottom;
    cv::Mat dashboard;
    cv::hconcat(original_panel, gray_panel, top);
    cv::hconcat(morph_panel, detection_panel, bottom);
    cv::vconcat(top, bottom, dashboard);
    return dashboard;
}

