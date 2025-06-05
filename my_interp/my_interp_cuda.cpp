/********************************************************************
 * my_interp_cuda.cpp  ―  Python binding
 *******************************************************************/
#include <torch/extension.h>

// CUDA 함수
torch::Tensor interp_radius20_launcher(const torch::Tensor& input,
                                       const torch::Tensor& coords);

torch::Tensor run(torch::Tensor input,
                  torch::Tensor coords)
{
    TORCH_CHECK(input.dim()==4 && input.size(3)==2, "input: [bs,10,125,2]");
    TORCH_CHECK(coords.dim()==2 && coords.size(1)==2, "coords: [N,2]");
    return interp_radius20_launcher(input.contiguous(), coords.contiguous());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "radius-20 interpolation (grid=batch×lane)");
}
