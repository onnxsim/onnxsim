// Execute the installed vendor NN runtime; the host only calculates a
// reference.
#include "installed_nn_abi.h"
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define LOAD(n)                                                                \
  __typeof__(&hexagon_nn_domains_##n) n =                                      \
      dlsym(lib, "hexagon_nn_domains_" #n);                                    \
  if (!n) {                                                                    \
    printf("missing %s\n", #n);                                                \
    return 2;                                                                  \
  }
#define CHECK(call)                                                            \
  do {                                                                         \
    err = (call);                                                              \
    if (err) {                                                                 \
      printf("FAIL %s: 0x%x\n", #call, err);                                   \
      goto cleanup;                                                            \
    }                                                                          \
  } while (0)
int main(int argc, char **argv) {
  void *lib = dlopen(argc > 1 ? argv[1] : "/vendor/lib64/libhexagon_nn_stub.so",
                     RTLD_NOW);
  if (!lib) {
    puts(dlerror());
    return 1;
  }
  LOAD(open);

  LOAD(close);
  LOAD(config);
  LOAD(init);
  LOAD(append_node);
  LOAD(append_const_node);

  LOAD(op_name_to_id);
  LOAD(prepare);
  LOAD(execute);
  LOAD(teardown);

  LOAD(getlog);
  remote_handle64 h = 0;
  int graph = 0, err = 0, failed = 1;
  CHECK(open(argc > 2 ? argv[2]
                      : "file:///"
                        "libhexagon_nn_skel.so?hexagon_nn_domains_skel_handle_"
                        "invoke&_modver=1.0&_dom=cdsp",
             &h));
  CHECK(config(h));
  CHECK(init(h, &graph));
  unsigned op_input, op_matmul, op_add, op_relu, op_output;
  CHECK(op_name_to_id(h, "INPUT", &op_input));
  CHECK(op_name_to_id(h, "MatMul_f", &op_matmul));
  CHECK(op_name_to_id(h, "Add_f", &op_add));
  CHECK(op_name_to_id(h, "Relu_f", &op_relu));
  CHECK(op_name_to_id(h, "OUTPUT", &op_output));
  const float weights[12] = {1, -2, 0.5f, -1, 1, 2, 0.5f, 0, -1, 2, 0.25f, 1};
  const float bias[3] = {0.25f, -0.5f, 1};
  hexagon_nn_output in_desc = {
      .rank = 4, .max_sizes = {1, 1, 1, 4}, .elementsize = 4};
  hexagon_nn_output out_desc = {
      .rank = 4, .max_sizes = {1, 1, 1, 3}, .elementsize = 4};
  CHECK(append_node(h, graph, 1, op_input, NN_PAD_NA, NULL, 0, &in_desc, 1));
  CHECK(append_const_node(h, graph, 2, 1, 1, 4, 3,
                          (const unsigned char *)weights, sizeof(weights)));
  CHECK(append_const_node(h, graph, 3, 1, 1, 1, 3, (const unsigned char *)bias,
                          sizeof(bias)));
  hexagon_nn_input mm_inputs[2] = {{1, 0}, {2, 0}},
                   add_inputs[2] = {{4, 0}, {3, 0}}, relu_inputs[1] = {{5, 0}},
                   output_inputs[1] = {{6, 0}};
  CHECK(append_node(h, graph, 4, op_matmul, NN_PAD_NA, mm_inputs, 2, &out_desc,
                    1));
  CHECK(
      append_node(h, graph, 5, op_add, NN_PAD_NA, add_inputs, 2, &out_desc, 1));
  CHECK(append_node(h, graph, 6, op_relu, NN_PAD_NA, relu_inputs, 1, &out_desc,
                    1));
  CHECK(append_node(h, graph, 7, op_output, NN_PAD_NA, output_inputs, 1, NULL,
                    0));
  CHECK(prepare(h, graph));
  const float cases[4][4] = {
      {1, 2, 3, 4}, {-1, 0, 2, -3}, {0, 0, 0, 0}, {3, -2, 0.5f, 1}};
  for (int c = 0; c < 4; c++) {
    float out[3] = {NAN, NAN, NAN};
    unsigned b = 0, ht = 0, w = 0, d = 0, bytes = 0;
    CHECK(execute(h, graph, 1, 1, 1, 4, (const unsigned char *)cases[c],
                  sizeof(cases[c]), &b, &ht, &w, &d, (unsigned char *)out,
                  sizeof(out), &bytes));
    if (b != 1 || ht != 1 || w != 1 || d != 3 || bytes != 12) {
      printf("FAIL output shape %ux%ux%ux%u bytes=%u\n", b, ht, w, d, bytes);
      goto cleanup;
    }
    printf("case %d output:", c);
    for (int j = 0; j < 3; j++) {
      float expected = bias[j];
      for (int k = 0; k < 4; k++)
        expected += cases[c][k] * weights[k * 3 + j];
      expected = fmaxf(0, expected);
      printf(" %g(expected %g)", out[j], expected);
      if (!isfinite(out[j]) || fabsf(out[j] - expected) > 1e-6f) {
        puts(" FAIL");
        goto cleanup;
      }
    }
    puts(" PASS");
  }

  failed = 0;
cleanup:
  if (graph) {
    if (failed) {
      unsigned char log[8192] = {0};
      getlog(h, graph, log, sizeof(log) - 1);
      printf("DSP graph log: %s\n", log);
    }
    int cleanup_error = teardown(h, graph);
    printf("teardown: 0x%x\n", cleanup_error);
    if (cleanup_error)
      failed = 1;
  }
  if (h && close(h))
    failed = 1;
  dlclose(lib);
  if (!failed)
    puts("PASS: 4 dense+ReLU inferences, all 12 output values match CPU "
         "reference");
  return failed;
}
