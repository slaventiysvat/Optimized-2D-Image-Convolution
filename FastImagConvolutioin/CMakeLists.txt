# CMakeList.txt: проект CMake для FastImagConvolutioin; включите исходный код и определения,
# укажите здесь логику для конкретного проекта.
#
cmake_minimum_required (VERSION 3.8)

project ("FastImagConvolutioin" VERSION 1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)


set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ImageConvolution/inc)

# Source files
set(SOURCE_FILES
    FastImagConvolutioin.cpp
    "ImageConvolution/src/reference.cpp"
)

# Добавьте источник в исполняемый файл этого проекта.
add_executable (FastImagConvolutioin ${PROJECT_NAME} ${SOURCE_FILES})

# TODO: Добавьте тесты и целевые объекты, если это необходимо.
