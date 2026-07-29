set(TARGET llama-epd)
add_executable(${TARGET} llama-epd.cpp llama-ep-transport.cpp llama-ep-transport.h)
target_link_libraries(${TARGET} PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
