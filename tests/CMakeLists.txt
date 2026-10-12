set(TCC_UNIT_TESTS
    test_main.cpp
    test_config.cpp
    test_indexing.cpp
    test_colormap.cpp
    test_state_io.cpp
    test_png.cpp
    test_cpu_solver.cpp
    test_legacy_equivalence.cpp
    test_setup.cpp
    test_field_guard.cpp
    test_stabilizer.cpp)

add_executable(tcc_unit_tests ${TCC_UNIT_TESTS})
target_link_libraries(tcc_unit_tests PRIVATE tcc_fluid)
target_include_directories(tcc_unit_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME unit COMMAND tcc_unit_tests)

if(TCC_HAVE_OPENCL)
    add_executable(tcc_gpu_tests test_main.cpp test_gpu_solver.cpp)
    target_link_libraries(tcc_gpu_tests PRIVATE tcc_fluid)
    target_include_directories(tcc_gpu_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME gpu COMMAND tcc_gpu_tests)
    # exit code 77 = no OpenCL device on this machine -> reported as skipped
    set_tests_properties(gpu PROPERTIES SKIP_RETURN_CODE 77)
endif()
