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
