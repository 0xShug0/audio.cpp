// Raw ggml quantization helper for the Python GGUF converters.
//
// Usage: ggml-quantize-raw <ggml-type-int> <row-size> <chunk-rows>
// Reads float32 rows from stdin and writes the quantized payload to stdout.
// The Python side (scripts/*/convert_*.py) uses this for tensor types that
// gguf-py cannot quantize natively, such as the K-quants.

#include "ggml.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

int main(int argc, char ** argv) {
#ifdef _WIN32
    if (_setmode(_fileno(stdin), _O_BINARY) == -1 ||
        _setmode(_fileno(stdout), _O_BINARY) == -1) {
        fprintf(stderr, "failed to switch standard streams to binary mode\n");
        return 1;
    }
#endif

    if (argc != 4) {
        fprintf(stderr, "usage: %s <ggml-type-int> <row-size> <chunk-rows>\n", argv[0]);
        return 1;
    }
    const enum ggml_type type = (enum ggml_type) atoi(argv[1]);
    const long long row_size = atoll(argv[2]);
    long long chunk_rows = atoll(argv[3]);
    if (row_size <= 0 || chunk_rows <= 0) {
        fprintf(stderr, "row-size and chunk-rows must be positive\n");
        return 1;
    }
    const size_t block_size = (size_t) ggml_blck_size(type);
    if (block_size == 0 || (size_t) row_size % block_size != 0) {
        fprintf(stderr, "row size %lld is not divisible by the %s block size\n",
                row_size, ggml_type_name(type));
        return 1;
    }
    const size_t row_bytes_out = ggml_row_size(type, row_size);

    float * input = malloc((size_t) chunk_rows * (size_t) row_size * sizeof(float));
    void * output = malloc((size_t) chunk_rows * row_bytes_out);
    if (input == NULL || output == NULL) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }

    for (;;) {
        const size_t values = fread(input, sizeof(float), (size_t) chunk_rows * (size_t) row_size, stdin);
        if (values == 0) {
            break;
        }
        if (values % (size_t) row_size != 0) {
            fprintf(stderr, "input is not a whole number of rows\n");
            return 1;
        }
        const long long rows = (long long) (values / (size_t) row_size);
        const size_t written = ggml_quantize_chunk(type, input, output, 0, rows, row_size, NULL);
        if (written != (size_t) rows * row_bytes_out) {
            fprintf(stderr, "quantize_chunk wrote %zu bytes, expected %zu\n",
                    written, (size_t) rows * row_bytes_out);
            return 1;
        }
        if (fwrite(output, 1, written, stdout) != written) {
            fprintf(stderr, "failed to write output\n");
            return 1;
        }
    }
    free(input);
    free(output);
    return 0;
}
