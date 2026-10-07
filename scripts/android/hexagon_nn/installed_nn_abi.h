// Minimal declarations for the legacy vendor Hexagon NN domains ABI.
// Layout follows the public nnlib interface/hexagon_nn.idl (see README).
// Functions are resolved with dlsym; no Qualcomm SDK libraries are linked.
#ifndef INSTALLED_NN_ABI_H
#define INSTALLED_NN_ABI_H
#include <stdint.h>
typedef uint64_t remote_handle64;
typedef int32_t hexagon_nn_nn_id;
typedef enum { NN_PAD_NA = 0 } hexagon_nn_padding_type;
typedef struct {
  uint32_t src_id, output_idx;
} hexagon_nn_input;
typedef struct {
  uint32_t rank, max_sizes[8], elementsize;
  int32_t zero_offset;
  float stepsize;
} hexagon_nn_output;
_Static_assert(sizeof(hexagon_nn_output) == 48, "NN output ABI size");
_Static_assert(sizeof(hexagon_nn_input) == 8, "NN input ABI size");
int hexagon_nn_domains_open(const char *uri, remote_handle64 *h);
int hexagon_nn_domains_close(remote_handle64 h);
int hexagon_nn_domains_config(remote_handle64 _h);
int hexagon_nn_domains_init(remote_handle64 _h, hexagon_nn_nn_id *g);
int hexagon_nn_domains_append_node(remote_handle64 _h, hexagon_nn_nn_id id,
                                   unsigned int node_id, unsigned int operation,
                                   hexagon_nn_padding_type padding,
                                   const hexagon_nn_input *inputs,
                                   int inputsLen,
                                   const hexagon_nn_output *outputs,
                                   int outputsLen);
int hexagon_nn_domains_append_const_node(
    remote_handle64 _h, hexagon_nn_nn_id id, unsigned int node_id,
    unsigned int batches, unsigned int height, unsigned int width,
    unsigned int depth, const unsigned char *data, int dataLen);
int hexagon_nn_domains_op_name_to_id(remote_handle64 _h, const char *name,
                                     unsigned int *node_id);
int hexagon_nn_domains_prepare(remote_handle64 _h, hexagon_nn_nn_id id);
int hexagon_nn_domains_execute(
    remote_handle64 _h, hexagon_nn_nn_id id, unsigned int batches_in,
    unsigned int height_in, unsigned int width_in, unsigned int depth_in,
    const unsigned char *data_in, int data_inLen, unsigned int *batches_out,
    unsigned int *height_out, unsigned int *width_out, unsigned int *depth_out,
    unsigned char *data_out, int data_outLen, unsigned int *data_len_out);
int hexagon_nn_domains_teardown(remote_handle64 _h, hexagon_nn_nn_id id);
int hexagon_nn_domains_getlog(remote_handle64 _h, hexagon_nn_nn_id id,
                              unsigned char *buf, int bufLen);
#endif
