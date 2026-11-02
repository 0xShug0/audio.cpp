#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-metal.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}
struct Shape {
    int w, h, d, ic, oc, n;
    int kw, kh, kd, sw, sh, sd, pw, ph, pd, dw, dh, dd;
    bool view = false;
};
struct Storage {
    ggml_context * ctx = ggml_init({8 * 1024 * 1024, nullptr, true});
    ggml_backend_buffer_t buffer = nullptr;
    ~Storage() {
        if (buffer) ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
};
void run(ggml_backend_t backend, Shape s, ggml_type type) {
    Storage storage;
    auto * ctx = storage.ctx;
    auto * xbase = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, s.w + (s.view ? 3 : 0), s.h + (s.view ? 1 : 0),
                                     s.d + (s.view ? 1 : 0), s.ic * s.n);
    auto * x = s.view ? ggml_view_4d(ctx, xbase, s.w, s.h, s.d, s.ic * s.n,
                                    xbase->nb[1], xbase->nb[2], xbase->nb[3], sizeof(float)) : xbase;
    auto * w = ggml_new_tensor_4d(ctx, type, s.kw, s.kh, s.kd, s.ic * s.oc);
    auto * packed = ggml_im2col_3d(ctx, w, x, s.ic, s.sw, s.sh, s.sd, s.pw, s.ph, s.pd,
                                  s.dw, s.dh, s.dd, type);
    // Independent gold: the existing Metal direct convolution. No CPU convolution
    // and no graph containing the new operation participates in this reference.
    auto * gold = ggml_conv_3d_direct(ctx, w, s.view ? ggml_cont(ctx, x) : x, s.sw, s.sh, s.sd, s.pw, s.ph, s.pd,
                                     s.dw, s.dh, s.dd, s.ic, s.n, s.oc);
    // Materialize views for the legacy convolution reference and the public
    // helper. The standalone packing comparison below still reads the raw view.
    auto * actual = ggml_conv_3d(ctx, w, s.view ? ggml_cont(ctx, x) : x, s.ic,
                                s.sw, s.sh, s.sd, s.pw, s.ph, s.pd, s.dw, s.dh, s.dd);
    // Check packing itself against existing Metal gather + cast, including zero
    // padding. The host builds gather indices, not simulated floating-point output.
    auto * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ggml_nelements(packed));
    auto * flat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, ggml_nelements(xbase) + 1);
    auto * gathered = ggml_get_rows(ctx, flat, indices);
    if (type == GGML_TYPE_F16) gathered = ggml_cast(ctx, gathered, type);
    auto * graph = ggml_new_graph_custom(ctx, 128, false);
    for (auto * out : {gold, actual, packed, gathered}) ggml_build_forward_expand(graph, out);
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto * node = ggml_graph_node(graph, i);
        if (!ggml_backend_supports_op(backend, node)) {
            std::cerr << "unsupported " << ggml_op_name(node->op) << " type=" << ggml_type_name(type) << " view=" << s.view << '\n';
            throw std::runtime_error("unsupported node");
        }
    }
    storage.buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(storage.buffer, "allocation failed");
    std::vector<int32_t> map(ggml_nelements(packed));
    size_t index = 0;
    const int od_count = packed->ne[3] / s.n;
    for (int n = 0; n < s.n; ++n)
    for (int od = 0; od < od_count; ++od)
    for (int oh = 0; oh < packed->ne[2]; ++oh)
    for (int ow = 0; ow < packed->ne[1]; ++ow)
    for (int ic = 0; ic < s.ic; ++ic)
    for (int kd = 0; kd < s.kd; ++kd)
    for (int kh = 0; kh < s.kh; ++kh)
    for (int kw = 0; kw < s.kw; ++kw) {
        int iw = ow*s.sw + kw*s.dw - s.pw, ih = oh*s.sh + kh*s.dh - s.ph, id = od*s.sd + kd*s.dd - s.pd;
        map[index++] = iw < 0 || iw >= s.w || ih < 0 || ih >= s.h || id < 0 || id >= s.d ?
            ggml_nelements(xbase) : (s.view ? 1 : 0) + iw +
            (ih*x->nb[1] + id*x->nb[2] + (n*s.ic+ic)*x->nb[3])/sizeof(float);
    }
    ggml_backend_tensor_set(indices, map.data(), 0, ggml_nbytes(indices));
    std::vector<float> first;
    for (int pattern = 0; pattern < 3; ++pattern) {
        std::vector<float> xv(ggml_nelements(xbase) + 1), wv(ggml_nelements(w));
        for (size_t i = 0; i + 1 < xv.size(); ++i)
            xv[i] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(float(int((i*17 + pattern*7)%31)-15) / (pattern ? 137.F : 16.F)));
        for (size_t i = 0; i < wv.size(); ++i)
            wv[i] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(float(int((i*13 + pattern*3)%23)-11) / (pattern ? 157.F : 32.F)));
        ggml_backend_tensor_set(xbase, xv.data(), 0, ggml_nbytes(xbase));
        ggml_backend_tensor_set(flat, xv.data(), 0, ggml_nbytes(flat));
        if (type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> half(wv.size());
            ggml_fp32_to_fp16_row(wv.data(), half.data(), half.size());
            ggml_backend_tensor_set(w, half.data(), 0, ggml_nbytes(w));
        } else ggml_backend_tensor_set(w, wv.data(), 0, ggml_nbytes(w));
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "compute failed");
        std::vector<unsigned char> p(ggml_nbytes(packed)), g(p.size());
        ggml_backend_tensor_get(packed, p.data(), 0, p.size());
        ggml_backend_tensor_get(gathered, g.data(), 0, g.size());
        require(p == g, "packing differs from legacy Metal gather/cast");
        std::vector<float> a(ggml_nelements(actual)), b(a.size());
        ggml_backend_tensor_get(actual, a.data(), 0, ggml_nbytes(actual));
        ggml_backend_tensor_get(gold, b.data(), 0, ggml_nbytes(gold));
        for (size_t i = 0; i < a.size(); ++i)
            require(std::isfinite(a[i]) && std::isfinite(b[i]) && std::abs(a[i]-b[i]) <= (pattern ? 2e-5F : 0.F),
                    "convolution differs from legacy Metal direct convolution");
        if (!pattern) first = a;
        else require(a != first, "updated input did not affect output");
    }
    // F32 packing must preserve values outside half range and half mantissa.
    if (type == GGML_TYPE_F32) {
        std::vector<float> xv(ggml_nelements(xbase) + 1);
        for (size_t i = 0; i + 1 < xv.size(); ++i) xv[i] = i%2 ? 65536.F : 1.0001F;
        ggml_backend_tensor_set(xbase, xv.data(), 0, ggml_nbytes(xbase));
        ggml_backend_tensor_set(flat, xv.data(), 0, ggml_nbytes(flat));
        auto * pack_graph = ggml_new_graph_custom(ctx, 128, false);
        ggml_build_forward_expand(pack_graph, packed);
        ggml_build_forward_expand(pack_graph, gathered);
        require(ggml_backend_graph_compute(backend, pack_graph) == GGML_STATUS_SUCCESS, "range compute failed");
        std::vector<float> a(ggml_nelements(packed)), b(a.size());
        ggml_backend_tensor_get(packed, a.data(), 0, ggml_nbytes(packed));
        ggml_backend_tensor_get(gathered, b.data(), 0, ggml_nbytes(gathered));
        require(a == b, "F32 packing lost range or precision");
    }
}
} // namespace
int main() {
    auto * backend = ggml_backend_metal_init();
    if (!backend) return 77;
    try {
        const Shape shapes[] = {
            {7,5,3,3,5,2, 1,1,1, 1,1,1, 0,0,0, 1,1,1},
            {7,5,4,3,5,2, 3,3,3, 1,1,1, 1,1,1, 1,1,1},
            {9,7,5,2,4,2, 2,3,2, 2,1,2, 1,2,1, 2,1,2},
            {5,4,1,3,2,1, 3,3,3, 1,1,1, 1,1,1, 1,1,1},
            {4,3,2,2,3,1, 2,2,2, 1,2,1, 0,0,0, 1,1,1},
            {9,7,5,2,4,2, 2,3,2, 2,1,2, 1,2,1, 2,1,2, true},
            {16,16,5,64,64,1, 3,3,3, 1,1,1, 1,1,1, 1,1,1},
            // More elements than a single bounded dispatch grid: exercise its loop.
            {32,32,9,128,8,1, 3,3,3, 1,1,1, 1,1,1, 1,1,1},
        };
        {
            Storage storage;
            auto * w = ggml_new_tensor_4d(storage.ctx, GGML_TYPE_F16, 3, 3, 3, 2);
            auto * x = ggml_new_tensor_4d(storage.ctx, GGML_TYPE_F32, 8, 8, 3, 2);
            auto * op = ggml_im2col_3d(storage.ctx, w, x, 2, 1,1,1, 1,1,1, 1,1,1, GGML_TYPE_F16);
            require(ggml_backend_supports_op(backend, op), "valid packing rejected");
            x->type = GGML_TYPE_F16;
            require(!ggml_backend_supports_op(backend, op), "unsupported input type admitted");
            x->type = GGML_TYPE_F32;
            x->nb[0] = 2*sizeof(float);
            require(!ggml_backend_supports_op(backend, op), "unsupported innermost stride admitted");
            x->nb[0] = sizeof(float);
            op->type = GGML_TYPE_I32;
            require(!ggml_backend_supports_op(backend, op), "unsupported output type admitted");
        }
        for (const auto & shape : shapes)
            for (auto type : {GGML_TYPE_F16, GGML_TYPE_F32}) run(backend, shape, type);
        std::cout << "PASS: 16 shapes/types, changing inputs, strided offset views, batches, stride/dilation/padding, legacy Metal gold\n";
        ggml_backend_free(backend);
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        ggml_backend_free(backend);
        return 1;
    }
}
