#include "armor_detector.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

struct Options {
    std::string video_path;
    std::string config_path = "config/detector.yaml";
    std::string output_video_path;
    std::string output_image_path;
    bool headless = false;
    int max_frames = 0;
};

void printUsage(const char* program) {
    std::cout
        << "Usage: " << program << " <video> [options]\n\n"
        << "Options:\n"
        << "  --config <path>        Detector YAML file\n"
        << "  --output-video <path> Save four-panel AVI/MP4 evidence\n"
        << "  --output-image <path> Save the best detection dashboard\n"
        << "  --max-frames <count>  Stop after N frames (0 = all)\n"
        << "  --headless            Do not open an imshow window\n"
        << "  --help                Show this help\n\n"
        << "Keys: Space pause/resume, Esc or Q quit.\n";
}

Options parseOptions(int argc, char** argv) {
    if (argc < 2) {
        printUsage(argv[0]);
        throw std::invalid_argument("Video path is required");
    }

    Options options;
    options.video_path = argv[1];
    for (int index = 2; index < argc; ++index) {
        const std::string argument = argv[index];
        auto nextValue = [&]() -> std::string {
            if (index + 1 >= argc) {
                throw std::invalid_argument("Missing value after " + argument);
            }
            return argv[++index];
        };

        if (argument == "--config") {
            options.config_path = nextValue();
        } else if (argument == "--output-video") {
            options.output_video_path = nextValue();
        } else if (argument == "--output-image") {
            options.output_image_path = nextValue();
        } else if (argument == "--max-frames") {
            options.max_frames = std::stoi(nextValue());
        } else if (argument == "--headless") {
            options.headless = true;
        } else if (argument == "--help") {
            printUsage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown option: " + argument);
        }
    }
    return options;
}

void createParentDirectory(const std::string& path) {
    if (path.empty()) {
        return;
    }
    const std::filesystem::path parent =
        std::filesystem::path(path).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        const DetectorConfig config =
            ArmorDetector::loadConfig(options.config_path);
        const ArmorDetector detector(config);

        cv::VideoCapture capture(options.video_path);
        if (!capture.isOpened()) {
            throw std::runtime_error(
                "Cannot open input video: " + options.video_path);
        }

        const double source_fps = capture.get(cv::CAP_PROP_FPS);
        const double output_fps = source_fps > 1.0 ? source_fps : 30.0;

        cv::VideoWriter writer;
        cv::Mat best_dashboard;
        std::size_t best_armor_count = 0;
        std::size_t best_light_count = 0;
        int best_light_quality = std::numeric_limits<int>::min();
        std::size_t frame_count = 0;
        std::size_t frames_with_lights = 0;
        std::size_t frames_with_armors = 0;
        double total_processing_ms = 0.0;
        bool paused = false;

        cv::Mat frame;
        while (capture.read(frame)) {
            const auto start = std::chrono::steady_clock::now();
            const DetectionResult result = detector.detect(frame);
            cv::Mat dashboard = makeDashboard(frame, result);
            const auto end = std::chrono::steady_clock::now();
            total_processing_ms +=
                std::chrono::duration<double, std::milli>(end - start).count();

            ++frame_count;
            if (!result.lights.empty()) {
                ++frames_with_lights;
            }
            if (!result.armors.empty()) {
                ++frames_with_armors;
            }

            // 仿真车正面通常能看到四根灯条。相同装甲板数量下，优先
            // 保存灯条数量最接近四根的画面，避免把偶发噪点当作最佳证据。
            const int light_quality = -std::abs(
                static_cast<int>(result.lights.size()) - 4);
            if (best_dashboard.empty() ||
                result.armors.size() > best_armor_count ||
                (result.armors.size() == best_armor_count &&
                 light_quality > best_light_quality)) {
                best_armor_count = result.armors.size();
                best_light_count = result.lights.size();
                best_light_quality = light_quality;
                best_dashboard = dashboard.clone();
            }

            if (!options.output_video_path.empty()) {
                if (!writer.isOpened()) {
                    createParentDirectory(options.output_video_path);
                    writer.open(options.output_video_path,
                                cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                output_fps, dashboard.size());
                    if (!writer.isOpened()) {
                        throw std::runtime_error(
                            "Cannot create output video: " +
                            options.output_video_path);
                    }
                }
                writer.write(dashboard);
            }

            if (!options.headless) {
                cv::imshow("Task 2 Armor Pipeline", dashboard);
                int key = cv::waitKey(paused ? 0 : 1);
                if (key == 27 || key == 'q' || key == 'Q') {
                    break;
                }
                if (key == ' ') {
                    paused = !paused;
                }
            }

            if (options.max_frames > 0 &&
                static_cast<int>(frame_count) >= options.max_frames) {
                break;
            }
        }

        if (frame_count == 0) {
            throw std::runtime_error("Input video contains no readable frames");
        }

        if (!options.output_image_path.empty()) {
            createParentDirectory(options.output_image_path);
            if (!cv::imwrite(options.output_image_path, best_dashboard)) {
                throw std::runtime_error(
                    "Cannot save output image: " + options.output_image_path);
            }
        }

        const double average_ms = total_processing_ms /
                                  static_cast<double>(frame_count);
        const double light_coverage =
            100.0 * static_cast<double>(frames_with_lights) /
            static_cast<double>(frame_count);
        const double armor_coverage =
            100.0 * static_cast<double>(frames_with_armors) /
            static_cast<double>(frame_count);
        std::cout << std::fixed << std::setprecision(2)
                  << "frames=" << frame_count << '\n'
                  << "frames_with_lights=" << frames_with_lights << '\n'
                  << "light_detection_coverage_pct=" << light_coverage << '\n'
                  << "frames_with_armors=" << frames_with_armors << '\n'
                  << "armor_detection_coverage_pct=" << armor_coverage << '\n'
                  << "best_light_count=" << best_light_count << '\n'
                  << "best_armor_count=" << best_armor_count << '\n'
                  << "average_processing_ms=" << average_ms << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}

