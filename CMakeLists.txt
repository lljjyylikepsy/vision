cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)
project(demo CXX)
find_package(OpenCV REQUIRED)

add_executable(main src/main.cpp)
target_link_libraries(main ${OpenCV_LIBS})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(task2 src/task2.cpp)
target_link_libraries(task2 ${OpenCV_LIBS})

add_executable(task3 src/task3.cpp)
target_link_libraries(task3 ${OpenCV_LIBS})

add_executable(task4 src/task4.cpp)
target_link_libraries(task4 ${OpenCV_LIBS})

add_executable(task5 src/task5.cpp)
target_link_libraries(task5 ${OpenCV_LIBS})

find_package(PkgConfig QUIET)
if (PkgConfig_FOUND)
    pkg_check_modules(TESSERACT QUIET tesseract)
endif()

add_executable(task6 src/task6.cpp)
target_link_libraries(task6 ${OpenCV_LIBS})

if (TESSERACT_FOUND)
    target_include_directories(task6 PRIVATE ${TESSERACT_INCLUDE_DIRS})
    target_link_libraries(task6 ${TESSERACT_LIBRARIES})
    target_compile_definitions(task6 PRIVATE HAVE_TESSERACT)
    message(STATUS "Tesseract found: ${TESSERACT_VERSION}")
else()
    message(WARNING "Tesseract not found, task6 will build without OCR")
endif()

add_executable(task7 src/task7.cpp)
target_link_libraries(task7 ${OpenCV_LIBS})

add_executable(task8 src/task8.cpp)
target_link_libraries(task8 ${OpenCV_LIBS})
