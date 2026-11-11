add_executable(softmax softmax_naive.cu)
add_executable(shuffle shuffle.cu)
target_link_libraries(softmax PRIVATE ${CUDA_LIBRARIES})
target_link_libraries(shuffle PRIVATE ${CUDA_LIBRARIES})