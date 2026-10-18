#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s in.gguf out.gguf [f16]\n", argv[0]); return 1; }
    const std::string target = argc > 3 ? argv[3] : "f16";
    ggml_context* tctx = nullptr;
    gguf_context* in = gguf_init_from_file(argv[1], {true, &tctx});
    if (!in || !tctx) { fprintf(stderr, "open failed\n"); return 1; }
    gguf_context* out = gguf_init_empty();
    gguf_set_kv(out, in);

    ggml_context* hctx = ggml_init({(size_t)200 << 20, nullptr, false});
    FILE* fp = fopen(argv[1], "rb");
    const size_t doff = gguf_get_data_offset(in);
    // Pass 1: add all tensor metadata (so the writer computes offsets), then
    // write meta only and append raw tensor data blocks in order.
    for (ggml_tensor* t = ggml_get_first_tensor(tctx); t; t = ggml_get_next_tensor(tctx, t)) {
        const std::string name = ggml_get_name(t);
        const bool is_weight = (name.size() > 7 && name.compare(name.size() - 7, 7, ".weight") == 0) && ggml_n_dims(t) >= 3;
        if (is_weight && t->type == GGML_TYPE_F32 && target == "f16") {
            ggml_tensor* nt = ggml_new_tensor(hctx, GGML_TYPE_F16, ggml_n_dims(t), t->ne);
            ggml_set_name(nt, name.c_str());
            gguf_add_tensor(out, nt);
        } else {
            gguf_add_tensor(out, t);
        }
    }
    if (!gguf_write_to_file(out, argv[2], true)) { fprintf(stderr, "meta write failed\n"); return 1; }
    // Pass 2: reopen the meta-only file to learn the computed data offset, then
    // write each tensor's bytes at data_offset + tensor_offset.
    {
        ggml_context* mc = nullptr;
        gguf_context* mg = gguf_init_from_file(argv[2], {true, &mc});
        if (!mg) { fprintf(stderr, "meta reopen failed\n"); return 1; }
        const size_t moff = gguf_get_data_offset(mg);
        gguf_free(mg); if (mc) ggml_free(mc);
        FILE* fm = fopen(argv[2], "r+b");
        if (!fm) { fprintf(stderr, "reopen failed\n"); return 1; }
        fseek(fm, 0, SEEK_END);
        const size_t fsize = ftell(fm);
        if (fsize < moff) { fprintf(stderr, "meta shorter than data offset\n"); return 1; }
        fclose(fm);   // data offset known; tensor writes below use absolute seeks
    }
    FILE* fo = fopen(argv[2], "r+b");
    if (!fo) { fprintf(stderr, "reopen failed\n"); return 1; }
    size_t n_conv = 0, n_keep = 0, bytes_before = 0, bytes_after = 0;
    std::vector<uint8_t> raw;
    std::vector<ggml_fp16_t> h16;
    for (ggml_tensor* t = ggml_get_first_tensor(tctx); t; t = ggml_get_next_tensor(tctx, t)) {
        const std::string name = ggml_get_name(t);
        const bool is_weight = (name.size() > 7 && name.compare(name.size() - 7, 7, ".weight") == 0) && ggml_n_dims(t) >= 3;
        const int64_t tid = gguf_find_tensor(in, name.c_str());
        const size_t off = gguf_get_tensor_offset(in, tid);
        const size_t nbytes = ggml_nbytes(t);
        raw.resize(nbytes);
        if (fseek(fp, doff + off, SEEK_SET) != 0 || fread(raw.data(), 1, nbytes, fp) != nbytes) {
            fprintf(stderr, "read failed for %s\n", name.c_str()); return 1;
        }
        bytes_before += nbytes;
        const int64_t tid_out = gguf_find_tensor(out, name.c_str());
        const size_t woff = gguf_get_tensor_offset(out, tid_out);
        static size_t data_off = 0; static bool off_init = false;
        if (!off_init) {
            // the meta-only file may be shorter than data_off (alignment gap not written)
            ggml_context* mc2 = nullptr;
            gguf_context* mg2 = gguf_init_from_file(argv[2], {true, &mc2});
            if (!mg2) { fprintf(stderr, "offset probe failed\n"); return 1; }
            data_off = gguf_get_data_offset(mg2);
            gguf_free(mg2); if (mc2) ggml_free(mc2);
            fprintf(stderr, "probe: data_off=%zu\n", data_off);
            off_init = true;
            fprintf(stderr, "data_off=%zu meta_file_size=%zu\n", data_off, fsize);
        }
        if (fseek(fo, data_off + woff, SEEK_SET) != 0) { fprintf(stderr, "seek failed\n"); return 1; }
        if (is_weight && t->type == GGML_TYPE_F32 && target == "f16") {
            const size_t n = nbytes / 4;
            h16.resize(n);
            const float* f = (const float*)raw.data();
            for (size_t i = 0; i < n; i++) h16[i] = ggml_fp32_to_fp16(f[i]);
            fwrite(h16.data(), 2, n, fo);
            n_conv++; bytes_after += n * 2;
        } else {
            fwrite(raw.data(), 1, nbytes, fo);
            n_keep++; bytes_after += nbytes;
        }
    }
    fclose(fo); fclose(fp);
    printf("converted %zu weight tensors to %s, kept %zu; %.1f MB -> %.1f MB\n",
           n_conv, target.c_str(), n_keep, bytes_before / 1e6, bytes_after / 1e6);
    gguf_free(out); gguf_free(in); ggml_free(tctx); ggml_free(hctx);
    return 0;
}
