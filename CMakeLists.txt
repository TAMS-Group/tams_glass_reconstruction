cmake_minimum_required(VERSION 2.9.3)
project(tams_glass)

set(CMAKE_CXX_FLAGS "-std=c++11")

set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  tf
  tf_conversions
  roscpp
  std_srvs
  visualization_msgs
  interactive_markers
  image_transport
  sensor_msgs
  cv_bridge
  eigen_conversions
  rosbag
  moveit_msgs
  moveit_ros_planning_interface
  image_geometry
  camera_info_manager
  camera_calibration_parsers
  pcl_conversions
  pcl_ros
)

catkin_package(
  CATKIN_DEPENDS
  roscpp
  tf
  tf_conversions
  sensor_msgs
  cv_bridge
  actionlib_msgs
  image_transport
  eigen_conversions
  rosbag
  moveit_msgs
  moveit_ros_planning_interface
  image_geometry
  camera_info_manager
  camera_calibration_parsers
  pcl_conversions
  pcl_ros
)

FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

find_package(Boost REQUIRED COMPONENTS program_options)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

add_executable(tams_glass_calibrate src/calibrate.cpp)
add_dependencies(tams_glass_calibrate ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(tams_glass_calibrate yaml-cpp ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS})
set_target_properties(tams_glass_calibrate PROPERTIES OUTPUT_NAME calibrate PREFIX "")

add_executable(tams_glass_reconstruct src/reconstruct.cpp)
add_dependencies(tams_glass_reconstruct ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(tams_glass_reconstruct yaml-cpp ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS})
set_target_properties(tams_glass_reconstruct PROPERTIES OUTPUT_NAME reconstruct PREFIX "")

add_executable(tams_glass_opaque src/opaque.cpp)
add_dependencies(tams_glass_opaque ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(tams_glass_opaque yaml-cpp ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS})
set_target_properties(tams_glass_opaque PROPERTIES OUTPUT_NAME opaque PREFIX "")

add_executable(tams_glass_poses src/poses.cpp)
add_dependencies(tams_glass_poses ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(tams_glass_poses yaml-cpp ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS})
set_target_properties(tams_glass_poses PROPERTIES OUTPUT_NAME poses PREFIX "")
