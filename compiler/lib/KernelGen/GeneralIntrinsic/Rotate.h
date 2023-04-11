#pragma once
#include "CvCommon.h"
#include "compiler/KernelGen/KernelGen.h"
namespace megcc {
namespace KernelGen {
namespace GeneralIntrinsic {

class RotateKernel : public CVKernelImpl {
public:
    bool IsCVAvailable(TContext* context) const override;
    std::string GetCVKernelBody(TContext* context) const override;
    std::string GetCVKernelSubSymbol(TContext* context) const override;
    std::string GetCVKernelSignature(TContext* context) const override;
};

}  // namespace GeneralIntrinsic
}  // namespace KernelGen
}  // namespace megcc