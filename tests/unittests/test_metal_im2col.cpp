#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-metal.h>

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef METAL_IM2COL_GOLDEN_PATH
#error "Define METAL_IM2COL_GOLDEN_PATH to the frozen baseline fixture"
#endif

namespace {

constexpr const char * kBaseline = "c8365f391307a97bfe133e0872e632fe572ecace";

struct Case {
    std::string name;
    int width, height, channels, batch, kw, kh, sx, sy, px, py, dx, dy;
    ggml_type type;
    bool is_2d = false;
    bool view = false;
    bool production = false;
    bool fast = false;
};

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::string sha256(const void * data, size_t size) {
    require(size <= UINT32_MAX, "digest input too large");
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(data, static_cast<CC_LONG>(size), digest);
    std::ostringstream out;
    for (unsigned char byte : digest) {
        out << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte);
    }
    return out.str();
}

std::vector<Case> make_cases() {
    std::vector<Case> cases;
    for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
        for (int width : {1, 17, 31, 32, 33, 63, 64, 65}) {
            for (int channels : {1, 3, 5}) {
                for (int kernel : {1, 3, 7, 11}) {
                    for (int stride : {1, 2}) {
                        for (int dilation : {1, 3}) {
                            for (int padding_mode : {0, 1}) {
                                if (kernel == 1 && padding_mode == 1) {
                                    continue;
                                }
                                const int padding = padding_mode * (kernel / 2) * dilation;
                                if (width + 2 * padding < dilation * (kernel - 1) + 1) {
                                    continue;
                                }
                                Case c{"boundary", width, 1, channels,
                                    (width + channels + stride) % 3 == 0 ? 3 : 1,
                                    kernel, 1, stride, 1, padding, 0, dilation, 1, type};
                                c.view = (width + channels + dilation) % 2 != 0;
                                cases.push_back(c);
                            }
                        }
                    }
                }
            }
        }
        for (int columns : {31, 32, 33, 63, 64, 65}) {
            for (int width : {31, 32, 33}) {
                cases.push_back({"patch_boundary", width, 1, columns, 3,
                    1, 1, 1, 1, 0, 0, 1, 1, type, false, true});
                cases.push_back({"tap_boundary", width, 1, 1, 1,
                    columns, 1, 1, 1, columns / 2, 0, 1, 1, type});
            }
        }
        for (int batch : {7, 257}) {
            cases.push_back({"batch_loop", 33, 1, 3, batch,
                7, 1, 2, 1, 3, 0, 1, 1, type, false, true});
        }
        for (int kernel : {2, 4, 32, 1023, 1024}) {
            cases.push_back({"large_or_even_kernel", 33, 1, 2, 3,
                kernel, 1, 1, 1, kernel / 2, 0, 1, 1, type});
        }
        cases.push_back({"padding_larger_than_input", 1, 1, 2, 3,
            3, 1, 2, 1, 17, 0, 5, 1, type});
        // The original 2D dispatch is retained by the PR, including batch iteration.
        for (int batch : {1, 3, 257}) {
            cases.push_back({"2d", 17, 9, 3, batch,
                3, 5, 2, 1, 2, 4, 2, 2, type, true, true});
            cases.push_back({"2d_pointwise", 33, 2, 5, batch,
                1, 1, 1, 1, 0, 0, 1, 1, type, true});
        }
        for (int kernel : {3, 7, 11}) {
            const int dilation = kernel == 11 ? 5 : 1;
            cases.push_back({"hift", 104281, 1, 64, 1,
                kernel, 1, 1, 1, kernel / 2 * dilation, 0, dilation, 1,
                type, false, false, true});
        }
        for (int width : {33, 65, 869}) {
            for (int kernel : {3, 7, 11}) {
                Case c{"fast_1d", width, 1, 5, 3,
                    kernel, 1, 2, 1, kernel / 2, 0, 1, 1, type};
                c.fast = true;
                cases.push_back(c);
            }
        }
    }
    return cases;
}

std::string case_id(const Case & c) {
    std::ostringstream out;
    out << c.name << '_' << ggml_type_name(c.type) << '_' << c.width << 'x' << c.height
        << "_c" << c.channels << "_n" << c.batch << "_k" << c.kw << 'x' << c.kh
        << "_s" << c.sx << 'x' << c.sy << "_p" << c.px << 'x' << c.py
        << "_d" << c.dx << 'x' << c.dy << "_view" << c.view;
    return out.str();
}

struct Graph {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ~Graph() {
        if (buffer) {
            ggml_backend_buffer_free(buffer);
        }
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

std::string run_case(ggml_backend_t backend, const Case & c, bool benchmark) {
    const std::string id = case_id(c);
    Graph resources;
    resources.ctx = ggml_init({2 * 1024 * 1024, nullptr, true});
    require(resources.ctx != nullptr, id + ": context allocation failed");
    auto * ctx = resources.ctx;
    // Metal advertises contiguous inputs; views vary the buffer offset, not row strides.
    const int channel_stride = c.width * c.height;
    const int batch_stride = channel_stride * c.channels;
    const int offset = c.view ? 9 : 0;
    std::vector<float> source(offset + size_t(batch_stride) * c.batch, -123.25f);
    for (int n = 0; n < c.batch; ++n) {
        for (int ch = 0; ch < c.channels; ++ch) {
            for (int y = 0; y < c.height; ++y) {
                for (int x = 0; x < c.width; ++x) {
                    const size_t index = offset + size_t(n) * batch_stride + ch * channel_stride + y * c.width + x;
                    // Integer-derived finite values are reproducible across compilers, including F16 ties.
                    uint32_t bits = uint32_t(index) * 1664525u + 1013904223u;
                    bits ^= bits >> 13;
                    float value = float(int(bits % 200001u) - 100000) / 2048.0f;
                    const float special[] = {0.0f, -0.0f, 65504.0f, -65504.0f,
                        1.00048828125f, 1.00146484375f, 0.00006103515625f};
                    if (index % 37 < sizeof(special) / sizeof(special[0])) {
                        value = special[index % 37];
                    }
                    source[index] = value;
                }
            }
        }
    }
    auto * storage = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, source.size());
    ggml_tensor * input;
    ggml_tensor * weight;
    if (c.is_2d) {
        input = ggml_view_4d(ctx, storage, c.width, c.height, c.channels, c.batch,
            c.width * sizeof(float), channel_stride * sizeof(float), batch_stride * sizeof(float), offset * sizeof(float));
        weight = ggml_new_tensor_4d(ctx, c.type, c.kw, c.kh, c.channels, 2);
    } else {
        input = ggml_view_3d(ctx, storage, c.width, c.channels, c.batch,
            channel_stride * sizeof(float), batch_stride * sizeof(float), offset * sizeof(float));
        weight = ggml_new_tensor_3d(ctx, c.type, c.kw, c.channels, 2);
    }
    ggml_tensor * output = nullptr;
    if (c.fast) {
        // Extract the actual lowering node built by the public convolution API.
        auto * convolution = ggml_conv_1d_fast_1d_im2col(ctx, weight, input, c.sx, c.px, c.dx);
        auto * convolution_graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(convolution_graph, convolution);
        for (int i = 0; i < ggml_graph_n_nodes(convolution_graph); ++i) {
            auto * node = ggml_graph_node(convolution_graph, i);
            if (node->op == GGML_OP_IM2COL_FAST_1D) {
                output = node;
                break;
            }
        }
        require(output != nullptr, id + ": fast convolution did not produce IM2COL_FAST_1D");
    } else {
        output = ggml_im2col(ctx, weight, input, c.sx, c.sy, c.px, c.py, c.dx, c.dy, c.is_2d, c.type);
    }
    require(ggml_backend_supports_op(backend, output), id + ": unsupported op");
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    resources.buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(resources.buffer != nullptr, id + ": buffer allocation failed");
    ggml_backend_tensor_set(storage, source.data(), 0, source.size() * sizeof(float));
    const int ow = (c.width + 2 * c.px - c.dx * (c.kw - 1) - 1) / c.sx + 1;
    const int oh = c.is_2d ? (c.height + 2 * c.py - c.dy * (c.kh - 1) - 1) / c.sy + 1 : 1;
    const int columns = c.channels * c.kh * c.kw;
    const std::array<int64_t, 4> shape = {columns, ow, c.is_2d ? oh : c.batch, c.is_2d ? c.batch : 1};
    for (size_t axis = 0; axis < shape.size(); ++axis) {
        require(output->ne[axis] == shape[axis], id + ": output shape mismatch");
    }
    std::vector<uint8_t> actual(ggml_nbytes(output));
    auto compute = [&] {
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, id + ": compute failed");
        ggml_backend_synchronize(backend);
    };
    const std::string input_digest = sha256(source.data(), source.size() * sizeof(float));
    std::string digest;
    // Change input and poison output on reuse to expose stale reads and unwritten tails.
    for (int repeat = 0; repeat < 2; ++repeat) {
        if (repeat) {
            for (float & value : source) {
                value = -value;
            }
            ggml_backend_tensor_set(storage, source.data(), 0, source.size() * sizeof(float));
        }
        std::fill(actual.begin(), actual.end(), repeat ? 0xa5 : 0x5a);
        ggml_backend_tensor_set(output, actual.data(), 0, actual.size());
        compute();
        ggml_backend_tensor_get(output, actual.data(), 0, actual.size());
        size_t index = 0;
        for (int n = 0; n < c.batch; ++n) {
            for (int y = 0; y < oh; ++y) {
                for (int x = 0; x < ow; ++x) {
                    for (int ch = 0; ch < c.channels; ++ch) {
                        for (int ky = 0; ky < c.kh; ++ky) {
                            for (int kx = 0; kx < c.kw; ++kx, ++index) {
                                const int ix = x * c.sx - c.px + kx * c.dx;
                                const int iy = y * c.sy - c.py + ky * c.dy;
                                const float expected = ix >= 0 && ix < c.width && iy >= 0 && iy < c.height
                                    ? source[offset + size_t(n) * batch_stride + ch * channel_stride + iy * c.width + ix]
                                    : 0.0f;
                                const ggml_fp16_t half = ggml_fp32_to_fp16(expected);
                                const void * expected_bytes = c.type == GGML_TYPE_F32
                                    ? static_cast<const void *>(&expected) : static_cast<const void *>(&half);
                                const size_t element_size = ggml_type_size(c.type);
                                if (std::memcmp(actual.data() + index * element_size, expected_bytes, element_size) != 0) {
                                    std::ostringstream message;
                                    message << id << ": scalar mismatch at element " << index
                                        << " (batch=" << n << ", y=" << y << ", x=" << x
                                        << ", channel=" << ch << ", ky=" << ky << ", kx=" << kx
                                        << "), expected=" << expected << ", actual_bits=";
                                    for (size_t b = 0; b < element_size; ++b) {
                                        message << std::hex << std::setw(2) << std::setfill('0')
                                            << unsigned(actual[index * element_size + b]);
                                    }
                                    throw std::runtime_error(message.str());
                                }
                            }
                        }
                    }
                }
            }
        }
        if (repeat) {
            digest += ' ';
        }
        digest += sha256(actual.data(), actual.size());
    }
    if (benchmark && c.production) {
        std::array<double, 5> times;
        for (double & elapsed : times) {
            const auto start = std::chrono::steady_clock::now();
            compute();
            elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        }
        std::sort(times.begin(), times.end());
        std::cout << id << " median_ms=" << times[2] << '\n';
    }
    std::ostringstream record;
    record << id;
    for (auto extent : shape) {
        record << ' ' << extent;
    }
    record << ' ' << actual.size() << ' ' << input_digest << ' ' << digest;
    return record.str();
}

int run(ggml_backend_t backend, int argc, char ** argv) {
    bool benchmark = false;
    bool capture = false;
    std::string path = METAL_IM2COL_GOLDEN_PATH;
    if (argc == 2 && std::string(argv[1]) == "--benchmark") {
        benchmark = true;
#ifdef METAL_IM2COL_CAPTURE_BASELINE
    } else if (argc == 3 && std::string(argv[1]) == "--capture-baseline") {
        capture = true;
        path = argv[2];
#endif
    } else {
        require(argc == 1, "usage: metal_im2col_test [--benchmark]");
    }
    std::map<std::string, std::string> golden;
    const std::string header = std::string("# metal_im2col_v1 baseline=") + kBaseline;
    std::ofstream captured;
    if (capture) {
        captured.open(path);
        require(bool(captured), "cannot create baseline fixture: " + path);
        captured << header << '\n';
    } else {
        std::ifstream fixture(path);
        require(bool(fixture), "missing frozen baseline fixture: " + path);
        std::string line;
        require(bool(std::getline(fixture, line)) && line == header, "incorrect baseline fixture provenance");
        while (std::getline(fixture, line)) {
            const auto end = line.find(' ');
            require(end != std::string::npos && golden.emplace(line.substr(0, end), line).second,
                "invalid or duplicate fixture record");
        }
        require(fixture.eof(), "failed to read baseline fixture");
    }
    const auto cases = make_cases();
    size_t production = 0;
    for (const auto & c : cases) {
        const std::string record = run_case(backend, c, benchmark);
        if (capture) {
            captured << record << '\n';
        } else {
            const auto found = golden.find(case_id(c));
            require(found != golden.end(), "missing baseline case: " + case_id(c));
            require(found->second == record, "frozen original-Metal output mismatch: " + case_id(c) +
                "\nexpected: " + found->second + "\nactual:   " + record);
            golden.erase(found);
        }
        production += c.production;
    }
    require(golden.empty(), "fixture contains cases that were not executed");
    if (capture) {
        captured.close();
        require(bool(captured), "failed to write baseline fixture");
    }
    std::cout << "PASS: " << cases.size() << " cases (" << production << " production), two executions each; "
        << (capture ? "captured original Metal output SHA-256 and " : "frozen original Metal SHA-256 and ")
        << "bit-exact scalar checks\n";
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    auto backend = ggml_backend_metal_init();
    if (!backend) {
        std::cout << "SKIP: Metal device unavailable\n";
        return 77;
    }
    int result = 1;
    try {
        result = run(backend, argc, argv);
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << '\n';
    }
    ggml_backend_free(backend);
    return result;
}
