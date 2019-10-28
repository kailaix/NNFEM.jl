cmake_minimum_required(VERSION 3.5)
project(AD)

set (CMAKE_CXX_STANDARD 11)

include_directories(Adept-2/include)
link_directories(Adept-2/adept/.libs)
add_library(nnlaw SHARED nnconstitutivelaw.cpp)
set_property(TARGET nnlaw PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(nnlaw adept)


