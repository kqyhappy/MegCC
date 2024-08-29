#include "Webassembly/KernelPack.h"
#include <memory>
#include "InternalKernel/InternalKernel.h"
#include "MatMulKernel/Fp32MatMul.h"

using namespace megcc;
using namespace KernelGen;
using namespace WebAssembly;

struct AllA32Kernel {
    AllA32Kernel() {
        inner_map[KernelPack::KernType::MatrixMulKernel] = {
                std::make_shared<WebAssembly::Fp32MatMulM4N12K4>()};

        inner_map[KernelPack::KernType::InternelKernel] = {
                std::make_shared<WebAssembly::MatmulM4N12MK4Kernel>()};
    }
    std::unordered_map<KernelPack::KernType, std::vector<std::shared_ptr<KernelFunc>>>
            inner_map;

};

std::vector<const KernelFunc*> WebAssembly::ArchKernelPack::GetKernel(
        KernelPack::KernType kernel_type) {
    static AllA32Kernel all_kernel;
    std::vector<const KernelFunc*> ret_kernels;
    for (auto& kernel : all_kernel.inner_map[kernel_type]) {
        ret_kernels.push_back(kernel.get());
    }
    return ret_kernels;
}