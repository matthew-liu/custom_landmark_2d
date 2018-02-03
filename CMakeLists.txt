cmake_minimum_required(VERSION 2.8.3)
project(custom_landmark_2d)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  pcl_ros
  image_geometry
  image_transport
  roscpp
  sensor_msgs
)

find_package( OpenCV REQUIRED )
find_package( PCL REQUIRED )

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES 
    matcher_2d
  CATKIN_DEPENDS
    cv_bridge
    pcl_ros
    image_geometry
    image_transport
    roscpp
    sensor_msgs
  DEPENDS
    OpenCV
    PCL
)

###########
## Build ##
###########


include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_library(custom_landmark_2d_matcher src/matcher.cpp)
add_dependencies(custom_landmark_2d_matcher ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(custom_landmark_2d_matcher 
  ${catkin_LIBRARIES}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_executable(custom_landmark_2d_demo src/demo_main.cpp)
set_target_properties(custom_landmark_2d_demo PROPERTIES OUTPUT_NAME demo PREFIX "")
target_link_libraries(custom_landmark_2d_demo
   custom_landmark_2d_matcher
   ${catkin_LIBRARIES}
)

add_executable(custom_landmark_2d_template_fetcher src/template_fetcher.cpp)
set_target_properties(custom_landmark_2d_template_fetcher PROPERTIES OUTPUT_NAME template_fetcher PREFIX "")
target_link_libraries(custom_landmark_2d_template_fetcher ${OpenCV_LIBS} ${catkin_LIBRARIES})

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
install(TARGETS 
  custom_landmark_2d_matcher
  custom_landmark_2d_demo
  custom_landmark_2d_template_fetcher
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
)

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_custom_landmark_2d.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)