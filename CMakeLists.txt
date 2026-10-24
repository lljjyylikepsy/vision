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

add_executable(task9 src/task9.cpp)
target_link_libraries(task9 ${OpenCV_LIBS})

add_executable(task10 src/task10.cpp)
target_link_libraries(task10 ${OpenCV_LIBS})

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/motor/control.cpp")
    enable_language(C)
    target_sources(task10 PRIVATE motor/control.cpp motor/angle_uart.c)
    target_include_directories(task10 PRIVATE motor)
    target_compile_definitions(task10 PRIVATE HAVE_MOTOR)
endif()

add_library(vision_core STATIC
    src/color.cpp
    src/shape.cpp
    src/digit.cpp
    src/arrow.cpp
    src/board.cpp
    src/calib.cpp
    src/target.cpp
    src/uart_sender.cpp
)
target_include_directories(vision_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(vision_core PUBLIC ${OpenCV_LIBS})

if(TESSERACT_FOUND)
    target_include_directories(vision_core PRIVATE ${TESSERACT_INCLUDE_DIRS})
    target_link_libraries(vision_core PRIVATE ${TESSERACT_LIBRARIES})
    target_compile_definitions(vision_core PRIVATE HAVE_TESSERACT)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/motor/control.cpp")
    if(NOT CMAKE_C_COMPILER_LOADED)
        enable_language(C)
    endif()
    target_sources(vision_core PRIVATE motor/control.cpp motor/angle_uart.c)
    target_include_directories(vision_core PRIVATE motor)
    target_compile_definitions(vision_core PRIVATE HAVE_MOTOR)
endif()

add_executable(task11 src/task11.cpp)
target_link_libraries(task11 PRIVATE vision_core)
