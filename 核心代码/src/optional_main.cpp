#include "optional_armor.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {

cv::Point2f armorCenter(const Pose& pose) {
    cv::Point2f center(0, 0);
    for (const auto& corner : pose.light_corners) {
        center += corner;
    }
    return center * 0.25f;
}

// 保持 CSV 列名及顺序稳定，独立评估脚本按此格式读取。
void writeDetectionRow(std::ostream& csv,
                        int frame_index,
                        std::size_t detection_index,
                        const Pose& pose,
                        const cv::Point2f& center) {
    csv << frame_index << ',' << detection_index << ',' << center.x << ',' << center.y
        << ',' << pose.translation_vector[0]
        << ',' << pose.translation_vector[1]
        << ',' << pose.translation_vector[2]
        << ',' << pose.distance_m << ',' << pose.reprojection_error_px;
    for (int axis = 0; axis < 3; ++axis) {
        csv << ',' << pose.rotation_vector[axis];
    }
    csv << ',' << pose.score;
    for (const auto& corner : pose.light_corners) {
        csv << ',' << corner.x << ',' << corner.y;
    }
    csv << '\n';
}

void drawArmorPose(cv::Mat& image,
                   const Pose& pose,
                   const Calibration& calibration,
                   const cv::Point2f& center) {
    for (int corner = 0; corner < 4; ++corner) {
        cv::line(image, pose.light_corners[corner], pose.light_corners[(corner + 1) % 4],
                 {0, 255, 0}, 2, cv::LINE_AA);
        cv::circle(image, pose.light_corners[corner], 3, {0, 255, 255}, -1);
    }
    cv::drawFrameAxes(image, calibration.camera_matrix, calibration.distortion_coefficients,
                      pose.rotation_vector, pose.translation_vector, 0.04f);

    std::ostringstream label;
    label << std::fixed << std::setprecision(3) << pose.distance_m << " m";
    cv::putText(image, label.str(), center + cv::Point2f(-30, -25),
                 cv::FONT_HERSHEY_SIMPLEX, 0.55, {255, 255, 255}, 1, cv::LINE_AA);
}

void drawFrameHeader(cv::Mat& image, int frame_index, std::size_t detection_count) {
    cv::rectangle(image, {0, 0, image.cols, 45}, {20, 20, 20}, -1);
    const std::string text = "Planar PnP | frame " + std::to_string(frame_index) +
                             " | detections " + std::to_string(detection_count);
    cv::putText(image, text, {15, 29}, cv::FONT_HERSHEY_SIMPLEX,
                 0.65, {255, 255, 255}, 1, cv::LINE_AA);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4) {
            std::cerr << "Usage: optional_armor input-video camera.yaml output-prefix\n";
            return 2;
        }

        // 1. 读取固定标定，打开输入视频，准备 CSV 和结果视频。
        const Calibration calibration = Calibration::load(argv[2]);
        cv::VideoCapture capture(argv[1]);
        if (!capture.isOpened()) {
            throw std::runtime_error("Cannot open input video");
        }

        const std::string output_prefix = argv[3];
        std::filesystem::create_directories(std::filesystem::path(output_prefix).parent_path());
        std::ofstream csv(output_prefix + "-detections.csv");
        csv << "frame,detection,cx,cy,x,y,z,distance,reprojection,r0,r1,r2,score,"
               "u0,v0,u1,v1,u2,v2,u3,v3\n";
        csv << std::setprecision(12);

        double fps = capture.get(cv::CAP_PROP_FPS);
        if (fps <= 0) {
            fps = 30;
        }
        cv::VideoWriter writer(
            output_prefix + "-result.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps,
            {calibration.image_width, calibration.image_height});
        if (!csv || !writer.isOpened()) {
            throw std::runtime_error("Cannot create output files");
        }

        // 2. 逐帧检测；输出使用观测结果，不用预测帧填补丢失。
        int frame_count = 0;
        int detected_frames = 0;
        double processing_time_ms = 0.0;
        cv::Mat image;
        while (capture.read(image)) {
            if (image.cols != calibration.image_width || image.rows != calibration.image_height) {
                throw std::runtime_error(
                    "Image size differs from calibration; update intrinsics after cropping/resizing");
            }

            const auto start = std::chrono::steady_clock::now();
            cv::Mat light_mask;
            const auto poses = detectOptionalArmor(image, calibration, light_mask);
            processing_time_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            if (!poses.empty()) {
                ++detected_frames;
            }

            for (std::size_t index = 0; index < poses.size(); ++index) {
                const Pose& pose = poses[index];
                const cv::Point2f center = armorCenter(pose);
                writeDetectionRow(csv, frame_count, index, pose, center);
                drawArmorPose(image, pose, calibration, center);
            }
            drawFrameHeader(image, frame_count, poses.size());
            writer.write(image);

            if (frame_count == 0 || frame_count == 60) {
                cv::imwrite(output_prefix + "-frame-" + std::to_string(frame_count) + ".png", image);
            }
            ++frame_count;
        }

        // 3. 汇总运行信息；检测是否正确由独立评估脚本判断。
        if (frame_count == 0) {
            throw std::runtime_error("No readable frames");
        }
        std::cout << "frames=" << frame_count
                  << "\ndetected_frames=" << detected_frames
                  << "\naverage_processing_ms=" << processing_time_ms / frame_count << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
