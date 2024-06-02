# CMakeList.txt: проект CMake для FastImagConvolutioin; включите исходный код и определения,
# укажите здесь логику для конкретного проекта.
#
cmake_minimum_required (VERSION 3.8)

project ("FastImagConvolutioin")

# Добавьте источник в исполняемый файл этого проекта.
add_executable (FastImagConvolutioin "FastImagConvolutioin.cpp" "FastImagConvolutioin.h")

# TODO: Добавьте тесты и целевые объекты, если это необходимо.
