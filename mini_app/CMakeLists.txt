cmake_minimum_required(VERSION 3.18)

# Set project name and required languages
project(MiniApp LANGUAGES C CXX Fortran)

# Ensure compilers are explicitly set
#set(CMAKE_Fortran_COMPILER ftn)
set(CMAKE_CXX_COMPILER hipcc)
#set(CMAKE_C_COMPILER cc)


# Find HIP and ROCm dependencies (ensure ROCM_PATH is set in env)
find_package(HIP REQUIRED)
find_package(rocblas REQUIRED)


set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -O3 -homp -hacc -eZ -hfp3 -s real64")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
if(DEFINED SAVE_TEMPS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -save-temps -gline-tables-only")
endif()
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -O3")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/run)
add_compile_definitions(USE_CHEMGEN_GPU)

if(DEFINED PARALLEL_LEVEL)
    add_compile_definitions(HIP_CHEM_v${PARALLEL_LEVEL})
endif()
foreach(flag IN ITEMS LDS NO_TRANSPOSE FUSE_EG_C)
    if(DEFINED ${flag} AND ${flag} EQUAL 1)
        add_compile_definitions(${flag})
    endif()
    unset(${flag} CACHE)
endforeach()
# Ensure Fortran and HIP are compiled separately before linking
if(DEFINED PARALLEL_LEVEL)
    add_library(coef_lib OBJECT src/modules/coef_m_v${PARALLEL_LEVEL}.f90)
    add_library(chemgen_lib OBJECT src/chemgen_m.f90 $<TARGET_OBJECTS:coef_lib>)
else()
    add_library(chemgen_lib OBJECT src/chemgen_m.f90)
endif()

add_library(main_lib OBJECT src/main.f90 src/getrates.f90 src/getrates_i.f90 $<TARGET_OBJECTS:chemgen_lib>)

# HIP Library (compiled separately)
if(DEFINED PARALLEL_LEVEL)
    add_library(hip_lib OBJECT src/getrates_hip_v${PARALLEL_LEVEL}.cpp)
    set_property(TARGET hip_lib PROPERTY HIP_ARCHITECTURES "gfx90a")
    target_include_directories(hip_lib PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
    ${HIP_INCLUDE_DIRS} 
    ${rocblas_INCLUDE_DIRS}
    )
endif()

# Create final executable
if(DEFINED PARALLEL_LEVEL)
    add_executable(mini_app $<TARGET_OBJECTS:main_lib> $<TARGET_OBJECTS:hip_lib> $<TARGET_OBJECTS:chemgen_lib> $<TARGET_OBJECTS:coef_lib>)
else()
    add_executable(mini_app $<TARGET_OBJECTS:main_lib> $<TARGET_OBJECTS:chemgen_lib>)
endif()
set_target_properties(mini_app PROPERTIES LINKER_LANGUAGE Fortran)


# Link all necessary libraries
if(DEFINED PARALLEL_LEVEL)
    target_link_libraries(mini_app PRIVATE 
    "-L$ENV{CRAY_LIBSCI_PREFIX_DIR}/lib -lsci_cray -lm"
    "-L${ROCM_PATH}/lib -lamdhip64 -lrocblas -lhipblas"
    )
else()
    target_link_libraries(mini_app PRIVATE 
    "-L$ENV{CRAY_LIBSCI_PREFIX_DIR}/lib -lsci_cray -lm"
    )
endif()
unset(PARALLEL_LEVEL CACHE)