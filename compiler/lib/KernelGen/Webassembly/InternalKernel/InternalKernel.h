#pragma once
#include <string>
#include "WebAssembly/KernelCommon.h"
#include "compiler/KernelGen/KernelGen.h"

namespace megcc {
namespace KernelGen {
namespace WebAssembly {

class MatmulM4N12MK4Kernel : public WebAssemblyMatmulInternal {
public:
    std::string GetKernelSymbol(TContext*) const override;

    std::string GetKernelBody(TContext*) const override;
 
    std::string GetPackAWorkspaceBody(TContext*) const override;

    std::string GetPackBWorkspaceBody(TContext*) const override;
};



}  // namespace WebAssembly
}  // namespace KernelGen
}  // namespace megcc