#Map solvers to Ints
module Linearsolver

@enum(LINEARSOLVER,
        L2R_LR = 0,
        L2R_L2LOSS_SVC_DUAL = 1,
        L2R_L2LOSS_SVC = 2,
        L2R_L1LOSS_SVC_DUAL = 3,
        MCSVM_CS = 4,
        L1R_L2LOSS_SVC = 5,
        L1R_LR = 6,
        L2R_LR_DUAL = 7,
        L2R_L2LOSS_SVR = 11,
        L2R_L2LOSS_SVR_DUAL = 12,
        L2R_L1LOSS_SVR_DUAL = 13)

end

module Kernel

# https://github.com/cjlin1/libsvm/blob/557d85749aaf0ca83fd229af0f00e4f4cb7be85c/svm.h#L26
@enum KERNEL Linear Polynomial RadialBasis Sigmoid Precomputed

end
