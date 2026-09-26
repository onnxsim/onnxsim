// Builds OpenCL C files with the phone's vendor compiler, one per argument, and reports OK/FAIL (exit 1 if any failed).
// export_cl.py --adreno runs it per kernel in a child process: Qualcomm's compiler can crash on a kernel and then fails
// every later build in the same process. CLOPT sets the build options.
#include "tg_cl_runner.h"

int main(int argc, char** argv) {
  tgcl::load_lib();
  cl_platform_id p;
  cl_device_id d;
  TGCL_CK(tgcl::clGetPlatformIDs(1, &p, nullptr));
  TGCL_CK(tgcl::clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &d, nullptr));
  cl_int e;
  cl_context c = tgcl::clCreateContext(nullptr, 1, &d, nullptr, nullptr, &e);
  TGCL_CK(e);
  const char* opt = getenv("CLOPT");
  int bad = 0;
  for (int i = 1; i < argc; i++) {
    std::string s = tgcl::read_file(argv[i]);
    const char* sp = s.c_str();
    cl_program pr = tgcl::clCreateProgramWithSource(c, 1, &sp, nullptr, &e);
    cl_int r = tgcl::clBuildProgram(pr, 1, &d, opt ? opt : "", nullptr, nullptr);
    size_t n = 0;
    tgcl::clGetProgramBuildInfo(pr, d, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
    std::string log(n, '\0');
    tgcl::clGetProgramBuildInfo(pr, d, CL_PROGRAM_BUILD_LOG, n, log.data(), nullptr);
    printf("%s: %s %s\n", argv[i], r == CL_SUCCESS ? "OK" : "FAIL", r == CL_SUCCESS ? "" : log.substr(0, 300).c_str());
    bad += r != CL_SUCCESS;
  }
  return bad ? 1 : 0;
}
