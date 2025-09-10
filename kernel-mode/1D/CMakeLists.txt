cmake_minimum_required (VERSION 3.12)
project(TestSNAP
  LANGUAGES CXX
  )

find_package(Kokkos REQUIRED)

# don't allow in-source builds
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  message(STATUS "Warning! Building from the source directory is not allow")
  message(STATUS "Remove 'CMakeCache.txt' and 'CMakeFiles' and build from a separate directory")
  message(ERROR "In-source build")
endif()

SET(MyTarget redn)

message(STATUS "CMAKE_SOURCE_DIR = ${CMAKE_SOURCE_DIR}")
FILE(GLOB sources
  ${CMAKE_SOURCE_DIR}/*.cpp
  ${CMAKE_SOURCE_DIR}/*.h
  )

ADD_EXECUTABLE(
  ${MyTarget}
  ${sources}
  )

TARGET_LINK_LIBRARIES(${MyTarget} Kokkos::kokkos)
