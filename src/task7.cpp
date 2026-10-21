#include "common.hpp"
#include <vector>
#include <string>
#include <algorithm>
#include <limits>

struct CalibParams {
    cv::Size board{9, 6};
    double square = 25.0;
    int frequency = 15;
};

struct CalibResult {
    bool valid = false;
    int imageCount = 0;
    double rms = 0.0;
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    std::vector<double> viewErrors;
};

std::vector<cv::Point3f> makeObjectPoints(cv::Size board, double square);
bool detectCorners(const cv::Mat& img, cv::Size board, std::vector<cv::Point2f>& corners);
CalibResult runCalibration(const std::vector<std::vector<cv::Point3f>>& objectPoints, const std::vector<std::vector<cv::Point2f>>& imagePoints, cv::Size imageSize);
void saveCalib(const std::string& path, const CalibResult& r, cv::Size imageSize, const CalibParams& p);
void printCalib(const CalibResult& r, cv::Size imageSize, const CalibParams& p);
std::vector<std::string> listImages(const std::string& dir);
void processDirectory(const std::string& dir, const CalibParams& p);

void processImage(const std::string& path, const CalibParams& p) {
    cv::Mat img;
    if (!common::loadImage(path, img)) return;
    std::vector<cv::Point2f> corners;
    if (!detectCorners(img, p.board, corners)) {
        std::cerr << "[error] chessboard not found: " << path << std::endl;
        return;
    }
    cv::Mat copy = img.clone();
    cv::drawChessboardCorners(copy, p.board, corners, true);
    auto out = common::outputPath("task7_result.png");
    cv::imwrite(out.string(), copy);
    std::cout << "corners = " << corners.size() << std::endl;
    std::cout << "saving to: " << out.string() << std::endl;
    cv::imshow("result", copy);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return;
}

void processStream(cv::VideoCapture& cap, const CalibParams& p) {
    if (!cap.isOpened()) return;
    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePoints;
    std::vector<cv::Point2f> lastCorners;
    std::vector<cv::Point3f> objectTemplate = makeObjectPoints(p.board, p.square);
    cv::Size imageSize;
    cv::Mat frame;
    int frameID = 0;
    int frameCount = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    int maxViews = frameCount > 0 ? std::numeric_limits<int>::max() : 40;
    while (1) {
        cap >> frame;
        if (frame.empty()) break;
        if (imageSize.empty()) imageSize = frame.size();
        if (frameID % p.frequency == 0 && static_cast<int>(objectPoints.size()) < maxViews) {
            std::vector<cv::Point2f> corners;
            if (detectCorners(frame, p.board, corners)) {
                double move = 0.0;
                if (lastCorners.size() == corners.size()) {
                    for (size_t i = 0; i < corners.size(); ++i) move += cv::norm(corners[i] - lastCorners[i]);
                    move /= static_cast<double>(corners.size());
                } else {
                    move = 1e9;
                }
                if (move > 10.0) {
                    objectPoints.push_back(objectTemplate);
                    imagePoints.push_back(corners);
                    lastCorners = corners;
                    std::cout << "captured view " << objectPoints.size() << std::endl;
                }
                cv::drawChessboardCorners(frame, p.board, corners, true);
            }
        }
        frameID++;
        cv::imshow("result", frame);
        if (cv::waitKey(30) == 27) break;
        if (frameCount <= 0 && static_cast<int>(objectPoints.size()) >= maxViews) break;
    }
    cap.release();
    cv::destroyAllWindows();
    if (imagePoints.size() < 3) {
        std::cerr << "[error] valid views < 3" << std::endl;
        return;
    }
    CalibResult r = runCalibration(objectPoints, imagePoints, imageSize);
    if (!r.valid) return;
    auto out = common::outputPath("calib.yml");
    saveCalib(out.string(), r, imageSize, p);
    printCalib(r, imageSize, p);
    return;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <image dir | image | video | camera index> <boardSizeCols> <boardSizeRows> <squareSize> <frequency>" << std::endl;
        return 1;
    }
    std::string path = argv[1];
    CalibParams p;
    if (argc >= 3) p.board.width = std::stoi(argv[2]);
    if (argc >= 4) p.board.height = std::stoi(argv[3]);
    if (argc >= 5) p.square = std::stod(argv[4]);
    if (argc >= 6) p.frequency = std::stoi(argv[5]);
    if (common::isCameraIndex(path)) {
        auto cap = common::openCamera(path);
        if (!cap.isOpened()) return 1;
        processStream(cap, p);
        return 0;
    }
    if (!common::ensurePathExists(path)) return 1;
    if (std::filesystem::is_directory(path)) {
        processDirectory(path, p);
    } else if (common::isVideoFile(path)) {
        auto cap = common::openVideo(path);
        processStream(cap, p);
    } else {
        processImage(path, p);
    }
    return 0;
}

std::vector<cv::Point3f> makeObjectPoints(cv::Size board, double square) {
    std::vector<cv::Point3f> pts;
    for (int y = 0; y < board.height; ++y)
        for (int x = 0; x < board.width; ++x)
            pts.emplace_back(x * square, y * square, 0.0f);
    return pts;
}

bool detectCorners(const cv::Mat& img, cv::Size board, std::vector<cv::Point2f>& corners) {
    static cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    cv::Mat gray, eq;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    if (cv::findChessboardCornersSB(gray, board, corners)) return true;
    clahe->apply(gray, eq);
    int strong = cv::CALIB_CB_EXHAUSTIVE | cv::CALIB_CB_ACCURACY;
    if (cv::findChessboardCornersSB(eq, board, corners, strong)) return true;
    if (cv::findChessboardCornersSB(gray, board, corners, strong)) return true;
    bool ok = cv::findChessboardCorners(eq, board, corners, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
    if (!ok) return false;
    cv::cornerSubPix(eq, corners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001));
    return true;
}

std::vector<std::string> listImages(const std::string& dir) {
    std::vector<std::string> files;
    for (const auto& e : std::filesystem::directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;
        std::string ext = e.path().extension().string();
        if (ext == ".png" || ext == ".PNG" || ext == ".jpg" || ext == ".JPG" || ext == ".jpeg" || ext == ".JPEG" || ext == ".bmp" || ext == ".BMP") files.push_back(e.path().string());
    }
    std::sort(files.begin(), files.end());
    return files;
}

CalibResult runCalibration(const std::vector<std::vector<cv::Point3f>>& objectPoints, const std::vector<std::vector<cv::Point2f>>& imagePoints, cv::Size imageSize) {
    CalibResult r;
    r.imageCount = static_cast<int>(objectPoints.size());
    if (r.imageCount < 3) return r;
    std::vector<cv::Mat> rvecs, tvecs;
    r.rms = cv::calibrateCamera(objectPoints, imagePoints, imageSize, r.cameraMatrix, r.distCoeffs, rvecs, tvecs, cv::CALIB_FIX_K3);
    r.valid = true;
    for (size_t i = 0; i < objectPoints.size(); ++i) {
        std::vector<cv::Point2f> projected;
        cv::projectPoints(objectPoints[i], rvecs[i], tvecs[i], r.cameraMatrix, r.distCoeffs, projected);
        r.viewErrors.push_back(cv::norm(imagePoints[i], projected, cv::NORM_L2) / static_cast<double>(projected.size()));
    }
    return r;
}

void saveCalib(const std::string& path, const CalibResult& r, cv::Size imageSize, const CalibParams& p) {
    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;
    fs << "board_width" << p.board.width;
    fs << "board_height" << p.board.height;
    fs << "square_size" << p.square;
    fs << "camera_matrix" << r.cameraMatrix;
    fs << "distortion_coefficients" << r.distCoeffs;
    fs << "rms" << r.rms;
    fs.release();
}

void printCalib(const CalibResult& r, cv::Size imageSize, const CalibParams& p) {
    std::cout << "image size: " << imageSize.width << "x" << imageSize.height << std::endl;
    std::cout << "board: " << p.board.width << "x" << p.board.height << "; square = " << p.square << "mm" << std::endl;
    std::cout << "valid views: " << r.imageCount << "; rms = " << r.rms << "px" << std::endl;
    std::cout << "camera matrix:\n" << r.cameraMatrix << std::endl;
    std::cout << "distortion coefficients:\n" << r.distCoeffs << std::endl;
    for (size_t i = 0; i < r.viewErrors.size(); ++i) std::cout << "view " << i << " error = " << r.viewErrors[i] << "px" << std::endl;
    return;
}

void processDirectory(const std::string& dir, const CalibParams& p) {
    std::vector<std::string> files = listImages(dir);
    if (files.empty()) {
        std::cerr << "[error] no images in " << dir << std::endl;
        return;
    }
    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePoints;
    std::vector<cv::Point3f> objectTemplate = makeObjectPoints(p.board, p.square);
    cv::Size imageSize;
    for (const auto& f : files) {
        cv::Mat img = cv::imread(f);
        if (img.empty()) continue;
        if (imageSize.empty()) imageSize = img.size();
        if (img.size() != imageSize) {
            std::cerr << "[warn] size mismatch, skip: " << f << std::endl;
            continue;
        }
        std::vector<cv::Point2f> corners;
        if (!detectCorners(img, p.board, corners)) {
            std::cerr << "[warn] chessboard not found: " << f << std::endl;
            continue;
        }
        objectPoints.push_back(objectTemplate);
        imagePoints.push_back(corners);
        std::cout << "captured view " << objectPoints.size() << ": " << f << std::endl;
    }
    if (imagePoints.size() < 3) {
        std::cerr << "[error] valid views < 3" << std::endl;
        return;
    }
    CalibResult r = runCalibration(objectPoints, imagePoints, imageSize);
    if (!r.valid) return;
    auto out = common::outputPath("calib.yml");
    saveCalib(out.string(), r, imageSize, p);
    printCalib(r, imageSize, p);
    return;
}
