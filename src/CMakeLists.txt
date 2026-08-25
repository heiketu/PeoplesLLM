llama_add_compile_flags()

#
# libraries
#

# llama

file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")

add_library(llama
            ../include/llama.h
            llama.cpp
            llama-adapter.cpp
            llama-arch.cpp
            llama-batch.cpp
            llama-chat.cpp
            llama-context.cpp
            llama-cparams.cpp
            llama-grammar.cpp
            llama-graph.cpp
            llama-hparams.cpp
            llama-hot-expert.cpp
            llama-impl.cpp
            llama-io.cpp
            llama-layer-major.cpp
            llama-kv-cache.cpp
            llama-kv-cache-iswa.cpp
            llama-kv-cache-dsa.cpp
            llama-kv-cache-msa.cpp
            llama-kv-cache-dsv4.cpp
            llama-memory.cpp
            llama-memory-hybrid.cpp
            llama-memory-hybrid-iswa.cpp
            llama-memory-recurrent.cpp
            llama-mmap.cpp
            llama-numa.cpp
            llama-model-loader.cpp
            llama-model-saver.cpp
            llama-model.cpp
            llama-quant.cpp
            llama-remote-ep.cpp
            ../tools/epd/llama-ep-protocol.cpp
            ../tools/epd/llama-ep-transport.cpp
            llama-sampler.cpp
            llama-vocab.cpp
            unicode-data.cpp
            unicode.cpp
            unicode.h
            ${LLAMA_MODELS_SOURCES}
            )

if (NOT LLAMA_USE_SYSTEM_GGML AND TARGET ggml-cpu AND NOT GGML_BACKEND_DL)
    target_sources(llama PRIVATE llama-repack-stream-bake.cpp)
    target_compile_definitions(llama PRIVATE LLAMA_REPACK_STREAM_BAKE)
endif()

set_target_properties(llama PROPERTIES
    VERSION ${LLAMA_VERSION_BASE}
    SOVERSION ${LLAMA_VERSION_MAJOR}
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
)

target_compile_definitions(llama PRIVATE
    LLAMA_VERSION="${LLAMA_VERSION}"
    LLAMA_COMMIT="${LLAMA_BUILD_COMMIT}"
)

target_include_directories(llama PRIVATE .)
target_include_directories(llama PUBLIC ../include)
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump

target_link_libraries(llama PUBLIC ggml)

if (LLAMA_EP_RDMA_ENABLED)
    target_sources(llama PRIVATE ../tools/epd/llama-ep-rdma.cpp)
    target_compile_definitions(llama PRIVATE LLAMA_EP_HAVE_RDMA)
    target_link_libraries(llama PRIVATE ${LLAMA_EP_IBVERBS_LIBRARY} ${LLAMA_EP_RDMACM_LIBRARY})
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()
