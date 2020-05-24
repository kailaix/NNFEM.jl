cmake_minimum_required(VERSION 3.5)
project(TFApp)
set (CMAKE_CXX_STANDARD 11)



execute_process(COMMAND julia -e "import ADCME; print(ADCME.__STR__)" OUTPUT_VARIABLE JL_OUT)
list(GET JL_OUT 0 BINDIR)
list(GET JL_OUT 1 LIBDIR)
list(GET JL_OUT 2 TF_INC)
list(GET JL_OUT 3 TF_ABI)
list(GET JL_OUT 4 EIGEN_INC)
list(GET JL_OUT 5 CC)
list(GET JL_OUT 6 CXX)
list(GET JL_OUT 7 CMAKE)
list(GET JL_OUT 8 MAKE)
list(GET JL_OUT 9 GIT)
list(GET JL_OUT 10 PYTHON)
list(GET JL_OUT 11 TF_LIB_FILE)

message("Python path=${PYTHON}")
message("EIGEN_INC=${EIGEN_INC}")
message("TF_INC=${TF_INC}")
message("TF_ABI=${TF_ABI}")
message("TF_LIB_FILE=${TF_LIB_FILE}")


if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0 OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 5.0)
  set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=${TF_ABI} ${CMAKE_CXX_FLAGS}")
endif()

# set(CMAKE_BUILD_TYPE Debug)
if(MSVC)
  if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "/W0" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W0")
  endif()
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
  add_definitions(-DNOMINMAX)
endif()

find_library(LIBOPENBLAS openblas HINTS ${LIBDIR})
set(CMAKE_BUILD_TYPE Release)

if(MSVC)
set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG")
else()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
endif()

include_directories(${TF_INC} ${HEADERS} ${EIGEN_INC} ${CMAKE_SOURCE_DIR}/DataStructure)
link_directories(${TF_LIB} ${CMAKE_SOURCE_DIR}/DataStructure/build)
include_directories(${LIBDIR}/Adept-2/include)
include_directories(${LIBDIR}/Adept-2/include)
find_library(ADEPT_LIB_FILE adept HINTS ${LIBDIR})
message("ADEPT_LIB_FILE=${ADEPT_LIB_FILE}")

set(LIBDIR_NAME CholOp CholOrthOp FintComp OrthotropicOp SPDOp SymOp DataStructure 
  SmallContinuumStrain SmallContinuumStiffness SmallContinuumFint FinitContinuumStrain FiniteContinuumFint
  Plasticity Isotropic TensorRep IsotropicTwo RivlinSaunders SmallContinuumStiffness1)
set(LIB_NAME CholOp CholOrthOp FintComp OrthotropicOp SPDOp SymOp data 
  SmallContinuumStrain SmallContinuumStiffness SmallContinuumFint FinitContinuumStrain FiniteContinuumFint
  Plasticity Isotropic TensorRep IsotropicTwo RivlinSaunders SmallContinuumStiffness1)


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
  set_target_properties(${_LIB_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/${_LIB_DIR}/build
          RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/${_LIB_DIR}/build  ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/${_LIB_DIR}/build)
  target_link_libraries(${_LIB_NAME} ${TF_LIB_FILE})
endforeach(IDX)

target_compile_definitions(data PRIVATE WIN_EXPORT)

target_link_libraries(SmallContinuumStrain data)
target_link_libraries(SmallContinuumStiffness data)
target_link_libraries(SmallContinuumStiffness1 data)
target_link_libraries(SmallContinuumFint data)
target_link_libraries(FinitContinuumStrain data)
target_link_libraries(FiniteContinuumFint data ${ADEPT_LIB_FILE} ${LIBOPENBLAS})
target_link_libraries(Plasticity ${ADEPT_LIB_FILE} ${LIBOPENBLAS})
target_link_libraries(IsotropicTwo ${ADEPT_LIB_FILE} ${LIBOPENBLAS})
target_link_libraries(RivlinSaunders ${ADEPT_LIB_FILE} ${LIBOPENBLAS})






