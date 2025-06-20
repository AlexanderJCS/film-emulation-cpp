cmake_minimum_required(VERSION 3.31)
project(film_emulation_cpp)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)
find_package(OpenColorIO REQUIRED)

find_package(OpenMP)

add_executable(film_emulation_cpp main.cpp
        imgio.cpp
        imgio.h
        grain_newson_et_al/film_grain_rendering.cpp
        grain_newson_et_al/film_grain_rendering.h
#        grain_newson_et_al/io_png.c
#        grain_newson_et_al/io_png.h
#        grain_newson_et_al/libtiff_io.cpp
#        grain_newson_et_al/libtiff_io.h
        grain_newson_et_al/matrix.cpp
        grain_newson_et_al/matrix.h
        grain_newson_et_al/pseudo_random_number_generator.cpp
        grain_newson_et_al/pseudo_random_number_generator.h
)

if(OpenMP_CXX_FOUND)
    target_compile_definitions(film_emulation_cpp PRIVATE USE_OPENMP)
    target_compile_options(film_emulation_cpp PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(film_emulation_cpp PRIVATE ${OpenMP_CXX_LIBRARIES})
else()
    message("Could not find OMP")
endif()


target_link_libraries(film_emulation_cpp
        PRIVATE
        ${OpenCV_LIBS}
        OpenColorIO::OpenColorIO
)