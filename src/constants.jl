#Map solvers to Ints
module Linearsolver
using Compat
@compat abstract type LinearSolver end
@compat abstract type L2R_LR<:LinearSolver end
@compat abstract type L2R_L2LOSS_SVC_DUAL<:LinearSolver end
@compat abstract type L2R_L2LOSS_SVC<:LinearSolver end
@compat abstract type L2R_L1LOSS_SVC_DUAL<:LinearSolver end
@compat abstract type MCSVM_CS<:LinearSolver end
@compat abstract type L1R_L2LOSS_SVC<:LinearSolver end
@compat abstract type L1R_LR<:LinearSolver end
@compat abstract type L2R_LR_DUAL<:LinearSolver end
@compat abstract type L2R_L2LOSS_SVR<:LinearSolver end
@compat abstract type L2R_L2LOSS_SVR_DUAL<:LinearSolver end
@compat abstract type L2R_L1LOSS_SVR_DUAL<:LinearSolver end

const SOLVERS = Dict{Type, Int32}(
        L2R_LR => 0,
        L2R_L2LOSS_SVC_DUAL => 1,
        L2R_L2LOSS_SVC => 2,
        L2R_L1LOSS_SVC_DUAL =>  3,
        MCSVM_CS => 4,
        L1R_L2LOSS_SVC => 5,
        L1R_LR => 6,
        L2R_LR_DUAL => 7,
        L2R_L2LOSS_SVR => 11,
        L2R_L2LOSS_SVR_DUAL => 12,
        L2R_L1LOSS_SVR_DUAL => 13
)
end

const SVMS = Dict{Symbol, Int32}(
    :CSVC => Int32(0),
    :nuSVC => Int32(1),
    :oneclassSVM => Int32(2),
    :epsilonSVR => Int32(3),
    :nuSVR => Int32(4)
    )

const KERNELS = Dict{Symbol, Int32}(
    :linear => Int32(0),
    :polynomial => Int32(1),
    :RBF => Int32(2),
    :sigmoid => Int32(3),
    :precomputed => Int32(4)
    )
