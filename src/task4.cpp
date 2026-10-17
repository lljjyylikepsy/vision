#include "common.hpp"
#include <vector>

int main(int argc, char** argv){
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " <image or video path>" << std::endl;
        return 1;
    }
    std::string path = argv[1];
    if (!std::filesystem::exists(path)) {
        std::cerr << "[error] path not exist: " << path << std::endl;
        return 1;
    }
    return 0;
}
