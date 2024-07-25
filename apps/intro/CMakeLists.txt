add_executable(intro)

set(SOURCE_LIST
    intro.cu	
)

target_sources(intro 
    PRIVATE
    ${SOURCE_LIST}
)

set_target_properties(intro PROPERTIES FOLDER "apps")

set_property(TARGET intro PROPERTY CUDA_SEPARABLE_COMPILATION ON)

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "intro" FILES ${SOURCE_LIST})

target_link_libraries(intro     
    PRIVATE RXMesh
)

#gtest_discover_tests( intro )