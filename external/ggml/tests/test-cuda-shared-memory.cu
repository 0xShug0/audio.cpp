#include "common.cuh"
#include "ggml-cpp.h"

#include <atomic>
#include <cmath>
#include <cstring>
#include <thread>

#ifdef GGML_TEST_VULKAN
#include <vulkan/vulkan.h>

struct vulkan_device {
    VkInstance instance = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;

    ~vulkan_device() {
        if (device) vkDestroyDevice(device, nullptr);
        if (instance) vkDestroyInstance(instance, nullptr);
    }

    int start() {
        VkApplicationInfo application{};
        application.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        application.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        info.pApplicationInfo = &application;
        const auto created = vkCreateInstance(&info, nullptr, &instance);
        if (created == VK_ERROR_INCOMPATIBLE_DRIVER) return 77;
        if (created != VK_SUCCESS) return 1;
        cudaDeviceProp cuda_properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&cuda_properties, 0));
        uint32_t count = 0;
        if (vkEnumeratePhysicalDevices(instance, &count, nullptr) != VK_SUCCESS) return 1;
        std::vector<VkPhysicalDevice> physical_devices(count);
        if (vkEnumeratePhysicalDevices(instance, &count, physical_devices.data()) != VK_SUCCESS) return 1;
        VkPhysicalDevice selected = VK_NULL_HANDLE;
        for (const auto candidate : physical_devices) {
            VkPhysicalDeviceIDProperties identity{};
            identity.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
            VkPhysicalDeviceProperties2 properties{};
            properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            properties.pNext = &identity;
            vkGetPhysicalDeviceProperties2(candidate, &properties);
            if (!std::memcmp(identity.deviceUUID, cuda_properties.uuid.bytes, VK_UUID_SIZE) &&
                properties.properties.apiVersion >= VK_API_VERSION_1_2) selected = candidate;
        }
        if (!selected) return 77;
        if (vkEnumerateDeviceExtensionProperties(selected, nullptr, &count, nullptr) != VK_SUCCESS) return 1;
        std::vector<VkExtensionProperties> extensions(count);
        if (vkEnumerateDeviceExtensionProperties(selected, nullptr, &count, extensions.data()) != VK_SUCCESS) return 1;
        const char * required[] = {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME};
        for (const auto name : required) {
            if (std::none_of(extensions.begin(), extensions.end(), [name](const auto & ext) {
                return std::strcmp(ext.extensionName, name) == 0;
            })) return 77;
        }
        VkPhysicalDeviceAccelerationStructureFeaturesKHR supported_as{};
        supported_as.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
        VkPhysicalDeviceVulkan12Features supported12{};
        supported12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        supported12.pNext = &supported_as;
        VkPhysicalDeviceFeatures2 supported{};
        supported.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        supported.pNext = &supported12;
        vkGetPhysicalDeviceFeatures2(selected, &supported);
        if (!supported12.bufferDeviceAddress || !supported_as.accelerationStructure) return 77;
        vkGetPhysicalDeviceQueueFamilyProperties(selected, &count, nullptr);
        std::vector<VkQueueFamilyProperties> families(count);
        vkGetPhysicalDeviceQueueFamilyProperties(selected, &count, families.data());
        uint32_t family = 0;
        while (family < count && !(families[family].queueFlags & VK_QUEUE_GRAPHICS_BIT)) ++family;
        if (family == count) return 77;
        const float priority = 1.0f;
        VkDeviceQueueCreateInfo queue{};
        queue.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue.queueFamilyIndex = family;
        queue.queueCount = 1;
        queue.pQueuePriorities = &priority;
        VkPhysicalDeviceAccelerationStructureFeaturesKHR acceleration{};
        acceleration.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
        acceleration.accelerationStructure = VK_TRUE;
        VkPhysicalDeviceVulkan12Features features{};
        features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        features.bufferDeviceAddress = VK_TRUE;
        features.pNext = &acceleration;
        VkDeviceCreateInfo device_info{};
        device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        device_info.pNext = &features;
        device_info.queueCreateInfoCount = 1;
        device_info.pQueueCreateInfos = &queue;
        device_info.enabledExtensionCount = 2;
        device_info.ppEnabledExtensionNames = required;
        const auto status = vkCreateDevice(selected, &device_info, nullptr, &device);
        if (status != VK_SUCCESS) std::fprintf(stderr, "vkCreateDevice failed: %d\n", status);
        return status == VK_SUCCESS ? 0 : 1;
    }
};
#endif

static __global__ void shared_probe(float * output, int count) {
    extern __shared__ float shared[];
    for (int i = threadIdx.x; i < count; i += blockDim.x) shared[i] = float(i);
    __syncthreads();
    for (int i = threadIdx.x; i < count; i += blockDim.x) output[i] = shared[count - i - 1];
}

static void launch_probe(float * output, int count, size_t bytes) {
    CUDA_SET_SHARED_MEMORY_LIMIT(shared_probe, bytes);
    shared_probe<<<1, 128, bytes>>>(output, count);
}

static bool check_probe(int bytes) {
    float * output = nullptr;
    CUDA_CHECK(cudaMalloc(&output, bytes));
    launch_probe(output, bytes / sizeof(float), bytes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> actual(bytes / sizeof(float));
    CUDA_CHECK(cudaMemcpy(actual.data(), output, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(output));
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != float(actual.size() - i - 1)) return false;
    }
    return true;
}

static CUcontext create_context() {
    CUcontext context = nullptr, popped = nullptr;
#if CUDA_VERSION >= 13000
    CU_CHECK(cuCtxCreate(&context, nullptr, 0, 0));
#else
    CU_CHECK(cuCtxCreate(&context, 0, 0));
#endif
    CU_CHECK(cuCtxPopCurrent(&popped));
    GGML_ASSERT(popped == context);
    return context;
}

static bool check_context_cache(int portable, int opt_in) {
    CUcontext original = nullptr;
    CU_CHECK(cuCtxGetCurrent(&original));
    GGML_ASSERT(original);
    if (!check_probe(opt_in)) return false;
    std::atomic<bool> smaller_success{false};
    std::thread smaller([&] {
        CU_CHECK(cuCtxSetCurrent(original));
        smaller_success.store(check_probe(portable));
        CU_CHECK(cuCtxSetCurrent(nullptr));
    });
    smaller.join();
    if (!smaller_success.load() || !check_probe(opt_in)) return false;
    CUcontext first = create_context(), second = create_context();
    for (const auto context : {first, second, first}) {
        CU_CHECK(cuCtxSetCurrent(context));
        for (const int bytes : {portable, opt_in, portable, opt_in}) {
            if (!check_probe(bytes)) return false;
        }
    }
    CU_CHECK(cuCtxSetCurrent(original));
    std::atomic<bool> success{true};
    auto worker = [&](CUcontext context) {
        CU_CHECK(cuCtxSetCurrent(context));
        if (!check_probe(opt_in) || !check_probe(portable)) success.store(false);
        CU_CHECK(cuCtxSetCurrent(nullptr));
    };
    std::thread left(worker, first), right(worker, second);
    left.join(); right.join();
    CU_CHECK(cuCtxDestroy(first));
    CU_CHECK(cuCtxDestroy(second));
    unsigned long long previous_id = 0;
    for (int repeat = 0; repeat < 3; ++repeat) {
        const auto context = create_context();
        CU_CHECK(cuCtxSetCurrent(context));
        unsigned long long id = 0;
        CU_CHECK(cuCtxGetId(context, &id));
        GGML_ASSERT(id != previous_id);
        previous_id = id;
        if (!check_probe(opt_in)) return false;
        CU_CHECK(cuCtxSetCurrent(original));
        CU_CHECK(cuCtxDestroy(context));
    }
    if (!check_probe(opt_in)) return false;
    std::puts("PASS full shared memory, context switching/recreation, growing requests and worker threads");
    return success.load();
}

static __global__ void static_probe(float * output) {
    __shared__ volatile float shared[256];
    shared[threadIdx.x] = float(threadIdx.x);
    __syncthreads();
    output[threadIdx.x] = shared[(threadIdx.x + 1) % 256];
}

static bool check_softmax(ggml_backend_t backend, int columns, ggml_type mask_type, bool use_mask = true) {
    const int rows = 2;
    const size_t elements = size_t(columns) * rows;
    ggml_init_params params = {ggml_tensor_overhead() * 4 + ggml_graph_overhead_custom(8, false), nullptr, true};
    ggml_context_ptr context(ggml_init(params));
    if (!context) return false;
    auto * input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, columns, rows);
    auto * mask = use_mask ? ggml_new_tensor_2d(context.get(), mask_type, columns, rows) : nullptr;
    auto * output = ggml_soft_max_ext(context.get(), input, mask, 1.0f, 0.0f);
    auto * graph = ggml_new_graph_custom(context.get(), 8, false);
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(context.get(), backend));
    if (!buffer) return false;
    std::vector<float> values(elements), mask_values(elements), actual(elements);
    std::vector<ggml_fp16_t> half_mask(elements);
    for (size_t i = 0; i < elements; ++i) {
        values[i] = std::sin(float(i) * 0.17f) * 3.0f;
        mask_values[i] = use_mask ? float(int(i % 7) - 3) * 0.25f : 0.0f;
        half_mask[i] = ggml_fp32_to_fp16(mask_values[i]);
    }
    ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
    if (mask) ggml_backend_tensor_set(mask, mask_type == GGML_TYPE_F32 ? static_cast<void *>(mask_values.data()) : half_mask.data(), 0, ggml_nbytes(mask));
    for (int repeat = 0; repeat < 3; ++repeat) {
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return false;
        ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
        for (int row = 0; row < rows; ++row) {
            const size_t offset = size_t(row) * columns;
            double maximum = -INFINITY;
            for (int col = 0; col < columns; ++col) maximum = std::max(maximum, double(values[offset + col] + mask_values[offset + col]));
            double sum = 0.0;
            for (int col = 0; col < columns; ++col) sum += std::exp(values[offset + col] + mask_values[offset + col] - maximum);
            for (int col = 0; col < columns; ++col) {
                const double expected = std::exp(values[offset + col] + mask_values[offset + col] - maximum) / sum;
                const double value = actual[offset + col];
                if (!std::isfinite(value) || std::abs(value - expected) > 1e-6 + expected * 1e-4) {
                    std::fprintf(stderr, "columns=%d mask=%s repeat=%d row=%d col=%d actual=%g expected=%g\n",
                                 columns, use_mask ? ggml_type_name(mask_type) : "none", repeat, row, col, value, expected);
                    return false;
                }
            }
        }
    }
    std::printf("PASS columns=%d mask=%s\n", columns, use_mask ? ggml_type_name(mask_type) : "none");
    return true;
}

int main(int argc, char ** argv) {
    const bool with_vulkan = argc == 2 && std::strcmp(argv[1], "--vulkan") == 0;
    const bool invalid = argc == 2 && std::strcmp(argv[1], "--invalid-static-limit") == 0;
    if (argc != 1 && !with_vulkan && !invalid) return 2;
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
    CUDA_CHECK(cudaSetDevice(0));
#ifdef GGML_TEST_VULKAN
    vulkan_device vulkan;
    if (with_vulkan) {
        const int status = vulkan.start();
        if (status != 0) return status;
    }
#else
    if (with_vulkan) return 77;
#endif
    int portable = 0, opt_in = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&portable, cudaDevAttrMaxSharedMemoryPerBlock, 0));
    CUDA_CHECK(cudaDeviceGetAttribute(&opt_in, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    if (invalid) {
        cudaFuncAttributes attributes{};
        CUDA_CHECK(cudaFuncGetAttributes(&attributes, static_probe));
        GGML_ASSERT(attributes.sharedSizeBytes == 1024);
        CUDA_SET_SHARED_MEMORY_LIMIT(static_probe, opt_in);
        std::fprintf(stderr, "invalid static-plus-dynamic limit was accepted\n");
        return 1;
    }
    if (!check_context_cache(portable, opt_in)) return 1;
    ggml_backend_ptr backend(ggml_backend_cuda_init(0));
    if (!backend) return 1;
    for (const auto type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
        for (const int columns : {32, 64, 128, 256, 512, 1024, 2048, 4096, 65, 513, portable / 4 - 32}) {
            if (!check_softmax(backend.get(), columns, type)) return 1;
        }
        if (opt_in >= portable + 256) {
            for (const int columns : {portable / 4, opt_in / 4 - 32, portable / 4 + 32, 65, opt_in / 4 + 32}) {
                if (!check_softmax(backend.get(), columns, type)) return 1;
            }
        }
    }
    if (opt_in >= portable + 256 && !check_softmax(backend.get(), opt_in / 4 - 32, GGML_TYPE_F32, false)) return 1;
    CUDA_CHECK(cudaGetLastError());
    return 0;
}
