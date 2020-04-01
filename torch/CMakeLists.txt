cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_executable(cpptorch cpptorch.C)
target_link_libraries(cpptorch "${TORCH_LIBRARIES}")
set_property(TARGET cpptorch PROPERTY CXX_STANDARD 14)
