add_executable(matmul matmul.cu)

target_link_libraries(matmul PRIVATE ${CUDA_LIBRARIES} cublas)
