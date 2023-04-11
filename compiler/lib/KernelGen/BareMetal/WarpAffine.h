#pragma once
#include <sstream>
#include <string>
#include "CvCommon.h"
#include "compiler/KernelGen/KernelGen.h"
namespace megcc {
namespace KernelGen {
namespace BareMetal {

class WarpAffineKernel : public CVKernelImpl {
public:
    bool IsAvailable(TContext* context) const override;
    std::string GetKernelSymbol(TContext* context) const override;
    std::string GetKernelBody(TContext* context) const override;

    bool IsCVAvailable(TContext* context) const override;
    std::string GetCVKernelBody(TContext* context) const override;
    std::string GetCVKernelSubSymbol(TContext* context) const override;
    std::string GetCVKernelSignature(TContext* context) const override;
};

}  // namespace BareMetal
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
