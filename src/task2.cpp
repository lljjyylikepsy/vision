#include "common.hpp"

void processImage(const std::string& path) {
    cv::Mat img;
    if (!common::loadImage(path, img)) return;
    double t0 = common::tick();
    //
    cv::Mat copy = img.clone();
    //
    double costMs = common::tickMs(t0);
    std::cout << "FrameID: 0; ts = 0ms; cost = " << costMs << "ms" << std::endl;
    auto out = common::outputPath("task2_result.png");
    cv::imwrite(out.string(), copy);
    std::cout << "saving to: " << out.string() << std::endl;
    cv::imshow("result", copy);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return;
}

void processStream(cv::VideoCapture& cap, const std::string& outName) {
    if (!cap.isOpened()) return;
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 25.0;
    cv::Size sz(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)), static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
    cv::VideoWriter writer;
    if (sz.width > 0 && sz.height > 0) {
        auto outPath = common::outputPath(outName);
        writer.open(outPath.string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, sz);
        if (writer.isOpened()) {
            std::cout << "saving to: " << outPath.string() << std::endl;
        }
    }
    cv::Mat frame;
    int frameID = 0;
    while (1) {
        cap >> frame;
        if (frame.empty()) break;
        double t0 = common::tick();
        cv::Mat copy = frame.clone();
        double costMs = common::tickMs(t0);
        std::cout << "FrameID: " << frameID << "; ts = " << cap.get(cv::CAP_PROP_POS_MSEC) << "ms; cost = " << costMs << "ms" << std::endl;
        frameID++;
        if (writer.isOpened()) writer.write(copy);
        cv::imshow("result", copy);
        if (cv::waitKey(30) == 27) break;
    }
    writer.release();
    cap.release();
    cv::destroyAllWindows();
    return;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <image or video path>" << std::endl;
        return 1;
    }
    std::string path = argv[1];
    if (!path.empty() && path.find_first_not_of("0123456789") == std::string::npos) {
        cv::VideoCapture cap(std::stoi(path));
        if (!cap.isOpened()) {
            std::cerr << "[error] cannot open camera " << path << std::endl;
            return 1;
        }
        processStream(cap, "task2_camera.avi");
        return 0;
    }
    if (!std::filesystem::exists(path)) {
        std::cerr << "[error] path not exist: " << path << std::endl;
        return 1;
    }
    std::string ext = std::filesystem::path(path).extension().string();
    if (ext == ".mp4" || ext == ".avi" || ext == ".mov" || ext == ".mkv") {
        auto cap = common::openVideo(path);
        processStream(cap, "task2_video.avi");
    } else {
        processImage(path);
    }
    return 0;
}