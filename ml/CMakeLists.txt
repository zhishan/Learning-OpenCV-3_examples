cmake_minimum_required(VERSION 3.5.1)
# Other Options
# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# 
project(opencv-ml)

# Options
set(CMAKE_BUILD_TYPE Debug)

include(../cmake/utility.cmake RESULT_VARIABLE UTITLITY_CMAKE)
if ("${UTITLITY_CMAKE}" STREQUAL "NOTFOUND")
    message(FATAL_ERROR "Unable to load utility.cmake")
endif()

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

brainstorm(test_gcd)

# 5
brainstorm(DecisionTree-on-boston-house-prices)
brainstorm(DecisionTree-on-mushroom)

# 8
brainstorm(SVM)

