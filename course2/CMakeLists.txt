add_executable(malloc_cpy malloc_cpy.cu)
target_link_libraries(malloc_cpy PRIVATE CUDA::cudart)
