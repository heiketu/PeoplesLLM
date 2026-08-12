set(TARGET llama-epd)
add_executable(${TARGET}
    llama-epd.cpp
    llama-ep-protocol.cpp
    llama-ep-protocol.h
    llama-epd-runtime.h
    llama-ep-session-manager.cpp
    llama-ep-session-manager.h
    llama-ep-transport.cpp
    llama-ep-transport.h)
target_link_libraries(${TARGET} PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-ep-transport-bench)
add_executable(${TARGET} ep-transport-bench.cpp llama-ep-transport.cpp llama-ep-transport.h)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-ep-dealer-test)
add_executable(${TARGET} ep-dealer-test.cpp llama-ep-dealer.h llama-ep-topology.h)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
if (BUILD_TESTING)
    add_test(NAME llama-ep-dealer-test COMMAND $<TARGET_FILE:llama-ep-dealer-test>)
endif()

set(TARGET llama-ep-topology-test)
add_executable(${TARGET} ep-topology-test.cpp llama-ep-topology.h llama-ep-capability.h)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
if (BUILD_TESTING)
    add_test(NAME llama-ep-topology-test COMMAND $<TARGET_FILE:llama-ep-topology-test>)
endif()

set(TARGET llama-shard-plan-test)
add_executable(${TARGET} ep-shard-plan-test.cpp ../../ggml/include/ggml-shard-plan.h)
target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/include)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
if (BUILD_TESTING)
    add_test(NAME llama-shard-plan-test COMMAND $<TARGET_FILE:llama-shard-plan-test>)
endif()

set(TARGET llama-ep-credit-test)
add_executable(${TARGET} ep-credit-test.cpp llama-ep-credit.h)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
if (BUILD_TESTING)
    add_test(NAME llama-ep-credit-test COMMAND $<TARGET_FILE:llama-ep-credit-test>)
endif()

set(TARGET llama-ep-autotune-test)
add_executable(${TARGET} ep-autotune-test.cpp llama-ep-autotune.h)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
if (BUILD_TESTING)
    add_test(NAME llama-ep-autotune-test COMMAND $<TARGET_FILE:llama-ep-autotune-test>)
endif()

set(TARGET llama-ep-service-test)
add_executable(${TARGET} ep-service-test.cpp llama-ep-service.h)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
if (BUILD_TESTING)
    add_test(NAME llama-ep-service-test COMMAND $<TARGET_FILE:llama-ep-service-test>)
endif()

set(TARGET llama-ep-fusion-test)
add_executable(${TARGET} ep-fusion-test.cpp llama-ep-fusion.h)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
if (BUILD_TESTING)
    add_test(NAME llama-ep-fusion-test COMMAND $<TARGET_FILE:llama-ep-fusion-test>)
endif()

set(TARGET llama-ep-parallel-test)
add_executable(${TARGET} ep-parallel-test.cpp llama-ep-parallel.h)
target_link_libraries(${TARGET} PRIVATE ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
if (BUILD_TESTING)
    add_test(NAME llama-ep-parallel-test COMMAND $<TARGET_FILE:llama-ep-parallel-test>)
endif()

set(TARGET llama-ep-session-manager-test)
add_executable(${TARGET}
    ep-session-manager-test.cpp
    llama-ep-session-manager.cpp
    llama-ep-session-manager.h)
target_link_libraries(${TARGET} PRIVATE ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
if (BUILD_TESTING)
    add_test(NAME llama-ep-session-manager-test COMMAND $<TARGET_FILE:llama-ep-session-manager-test>)
endif()

set(TARGET llama-ep-protocol-test)
add_executable(${TARGET}
    ep-protocol-test.cpp
    llama-ep-protocol.cpp
    llama-ep-protocol.h
    llama-ep-transport.cpp
    llama-ep-transport.h)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
if (BUILD_TESTING)
    add_test(NAME llama-ep-protocol-test COMMAND $<TARGET_FILE:llama-ep-protocol-test>)
endif()

set(TARGET llama-ep-expert-map-test)
add_executable(${TARGET} ep-expert-map-test.cpp llama-ep-expert-map.h)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
if (BUILD_TESTING)
    add_test(NAME llama-ep-expert-map-test COMMAND $<TARGET_FILE:llama-ep-expert-map-test>)
    find_program(LLAMA_EP_PYTHON3_EXECUTABLE NAMES python3)
    if (LLAMA_EP_PYTHON3_EXECUTABLE)
        add_test(NAME llama-ep-map-from-freq-test
            COMMAND ${LLAMA_EP_PYTHON3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/ep-map-from-freq-test.py)
    endif()
endif()

set(TARGET llama-ep-crossslot)
add_executable(${TARGET} ep-crossslot-bench.cpp)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
target_link_libraries(${TARGET} PRIVATE llama llama-common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (LLAMA_EP_RDMA_ENABLED)
    foreach (T llama-epd llama-ep-transport-bench)
        target_sources(${T} PRIVATE llama-ep-rdma.cpp)
        target_compile_definitions(${T} PRIVATE LLAMA_EP_HAVE_RDMA)
        target_link_libraries(${T} PRIVATE ${LLAMA_EP_IBVERBS_LIBRARY} ${LLAMA_EP_RDMACM_LIBRARY})
    endforeach()
endif()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS llama-epd RUNTIME)
endif()
