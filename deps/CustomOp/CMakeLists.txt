cmake_minimum_required(VERSION 3.5)
project(TFApp)

##  For PyTorch
# execute_process(COMMAND julia -e "abspath(joinpath(Base.find_package(\"ADCME\"), \"../../deps/Libraries/libtorch\"))|>print" OUTPUT_VARIABLE PREFIX)
# execute_process(COMMAND julia -e "abspath(joinpath(Base.find_package(\"ADCME\"), \"../../examples/custom_op/headers\"))|>print" OUTPUT_VARIABLE HEADERS)
# execute_process(COMMAND julia -e "abspath(joinpath(Base.find_package(\"ADCME\"), \"../../deps/Libraries\"))|>print" OUTPUT_VARIABLE EIGEN_INC)
# execute_process(COMMAND julia -e "abspath(joinpath(Base.find_package(\"ADCME\"), \"../../deps/Libraries/libtorch/include\"))|>print" OUTPUT_VARIABLE TORCH_INC)
# set(CMAKE_PREFIX_PATH ${PREFIX})
# find_package(Torch REQUIRED)

set (CMAKE_CXX_STANDARD 11)
# C++11 required for tensorflow
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

execute_process(COMMAND julia -e "using PyCall; print(PyCall.python)" OUTPUT_VARIABLE PYTHON)
message("Python path=${PYTHON}")

execute_process(COMMAND ${PYTHON} -c "import tensorflow as tf; import sys; sys.stdout.write(tf.sysconfig.get_compile_flags()[0][2:])" OUTPUT_VARIABLE TF_INC)
execute_process(COMMAND ${PYTHON} -c "import tensorflow as tf; import sys; sys.stdout.write(tf.sysconfig.get_link_flags()[0][2:])" OUTPUT_VARIABLE TF_LIB)
execute_process(COMMAND ${PYTHON} -c "import tensorflow as tf; import sys; sys.stdout.write(tf.sysconfig.get_compile_flags()[1][-1])" OUTPUT_VARIABLE TF_ABI)


# https://github.com/tensorflow/tensorflow/issues/1569
# if GCC > 5
if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0 OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 5.0)
  set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=${TF_ABI} ${CMAKE_CXX_FLAGS}")
endif()

include_directories(${TF_INC} ${HEADERS} ${EIGEN_INC} ${TORCH_INC})
link_directories(${TF_LIB})

set(LIBDIR_NAME CholOp CholOrthOp FintComp OrthotropicOp SPDOp SymOp)
set(LIB_NAME CholOp CholOrthOp FintComp OrthotropicOp SPDOp SymOp)

# Copy and paste the following lines (no modification is required)
list(LENGTH "LIBDIR_NAME" LIBLENGTH)
message("Total number of libraries to make: ${LIBLENGTH}")
MATH(EXPR LIBLENGTH "${LIBLENGTH}-1")
foreach(IDX RANGE 0 ${LIBLENGTH})
  list(GET LIBDIR_NAME ${IDX} _LIB_DIR)
  list(GET LIB_NAME ${IDX} _LIB_NAME)
  message("Compiling ${IDX}th library: ${_LIB_DIR}==>${_LIB_NAME}")
  file(MAKE_DIRECTORY ${_LIB_DIR}/build)
  add_library(${_LIB_NAME} SHARED ${_LIB_DIR}/${_LIB_NAME}.cpp)
  set_property(TARGET ${_LIB_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_target_properties(${_LIB_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/${_LIB_DIR}/build)
  target_link_libraries(${_LIB_NAME} ${TF_LIB_FILE})
endforeach(IDX)