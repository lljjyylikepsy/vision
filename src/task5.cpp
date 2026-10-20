#include "common.hpp"
#include <vector>

struct ShapeResult {
    int valid = 0;
    std::string label;
    cv::Point2f center = {0, 0};
    double area = 0.0;
};

std::string classifyShape(const std::vector<cv::Point>& contour);
std::vector<ShapeResult> detectShapes(const cv::Mat& img, double minArea);
void drawResults(cv::Mat& img, const std::vector<ShapeResult>& results);

void processImage(const std::string& path, double minArea) {
    cv::Mat img;
    if (!common::loadImage(path, img)) return;
    double t0 = common::tick();
    //
    std::vector<ShapeResult> results = detectShapes(img, minArea);
    //
    double costMs = common::tickMs(t0);
    std::cout << "shapes = " << results.size() << "; cost = " << costMs << "ms" << std::endl;
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << "[" << i << "] " << results[i].label << "; center=(" << results[i].center.x << "," << results[i].center.y << "); area=" << results[i].area << std::endl;
    }
    cv::Mat copy = img.clone();
    drawResults(copy, results);
    cv::imwrite(common::outputPath("task5_result.png").string(), copy);
    cv::imshow("result", copy);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return;
}

void processStream(cv::VideoCapture& cap, const std::string& outName, double minArea) {
    if (!cap.isOpened()) return;
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 25.0;
    cv::Size sz(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)), static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
    cv::VideoWriter writer;
    if (sz.width > 0 && sz.height > 0) {
        writer.open(common::outputPath(outName).string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, sz);
    }
    cv::Mat frame;
    int frameID = 0;
    while (1) {
        cap >> frame;
        if (frame.empty()) break;
        std::vector<ShapeResult> results = detectShapes(frame, minArea);
        std::cout << "FrameID: " << frameID << "; ts = " << cap.get(cv::CAP_PROP_POS_MSEC) << "ms; shapes = " << results.size() << std::endl;
        frameID++;
        drawResults(frame, results);
        if (writer.isOpened()) writer.write(frame);
        cv::imshow("result", frame);
        if (cv::waitKey(30) == 27) break;
    }
    writer.release();
    cap.release();
    cv::destroyAllWindows();
    return;
}

int main(int argc, char** argv){
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <image or video path>" << std::endl;
        return 1;
    }
    std::string path = argv[1];
    double minArea = 300.0;
    if (argc >= 3) minArea = std::stod(argv[2]);
    if (!path.empty() && path.find_first_not_of("0123456789") == std::string::npos) {
        cv::VideoCapture cap(std::stoi(path));
        if (!cap.isOpened()) {
            std::cerr << "[error] cannot open camera " << path << std::endl;
            return 1;
        }
        processStream(cap, "task5_camera.avi", minArea);
        return 0;
    }
    if (!std::filesystem::exists(path)) {
        std::cerr << "[error] path not exist: " << path << std::endl;
        return 1;
    }
    std::string ext = std::filesystem::path(path).extension().string();
    if (ext == ".mp4" || ext == ".avi" || ext == ".mov" || ext == ".mkv") {
        auto cap = common::openVideo(path);
        processStream(cap, "task5_video.avi", minArea);
    } else {
        processImage(path, minArea);
    }
    return 0;
}

std::string classifyShape(const std::vector<cv::Point>& contour) {
    double area = cv::contourArea(contour);
    double peri = cv::arcLength(contour, true);
    float radius = 0;
    cv::Point2f mc;
    cv::minEnclosingCircle(contour, mc, radius);
    double fill = (radius > 0) ? area / (CV_PI * radius * radius) : 0.0;
    std::vector<cv::Point> approx;
    cv::approxPolyDP(contour, approx, 0.02 * peri, true);
    int n = static_cast<int>(approx.size());
    if (fill > 0.85) return "circle";
    if (n == 3) return "triangle";
    if (n == 4) return "rectangle";
    return "unknown";
}

std::vector<ShapeResult> detectShapes(const cv::Mat& img, double minArea) {
    std::vector<ShapeResult> results;
    cv::Mat grey;
    cv::cvtColor(img, grey, cv::COLOR_BGR2GRAY);
    cv::Mat binary;
    cv::threshold(grey, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    int bw = 0;
    int bt = 0;
    for (int x = 0; x < binary.cols; ++x) {
        bw += binary.at<uchar>(0, x) > 0;
        bw += binary.at<uchar>(binary.rows - 1, x) > 0;
        bt += 2;
    }
    for (int y = 0; y < binary.rows; ++y) {
        bw += binary.at<uchar>(y, 0) > 0;
        bw += binary.at<uchar>(y, binary.cols - 1) > 0;
        bt += 2;
    }
    if (bw * 2 > bt) cv::bitwise_not(binary, binary);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (size_t i = 0; i < contours.size(); ++i) {
        double a = cv::contourArea(contours[i]);
        if (a < minArea) continue;
        ShapeResult r;
        r.label = classifyShape(contours[i]);
        cv::Moments m = cv::moments(contours[i]);
        if (m.m00 != 0) {
            r.center = cv::Point2f(static_cast<float>(m.m10 / m.m00), static_cast<float>(m.m01 / m.m00));
        }
        r.valid = 1;
        r.area = a;
        results.push_back(r);
    }
    return results;
}

void drawResults(cv::Mat& img, const std::vector<ShapeResult>& results) {
    for (size_t i = 0; i < results.size(); ++i) {
        cv::circle(img, cv::Point(static_cast<int>(results[i].center.x), static_cast<int>(results[i].center.y)), 5, cv::Scalar(0, 0, 255), -1);
        cv::putText(img, results[i].label, cv::Point(static_cast<int>(results[i].center.x) + 8, static_cast<int>(results[i].center.y) - 8), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    }
}