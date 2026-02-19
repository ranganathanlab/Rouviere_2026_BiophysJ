# This module defines functions to make figures for
# "Unification of Allosteric Models" 

module UnificationAllostery
using LinearAlgebra, SpecialFunctions, PyPlot, LaTeXStrings, Distributed,
    Random, StatsBase
import PyPlot.PyObject

push!(LOAD_PATH,"./")
include("quadratic.jl")
include("bistable.jl")
include("order_disorder.jl")
include("mcmc.jl")


export
    # from quadratic.jl
    prepareQuadSystem,
    freeEnergyQuad,
    coopQuad,
    confQuad,
    confChangeQuad,
    fluctQuad,
    fluctChangeQuad,
    
    # from bistable.jl
    freeEnergyBi,
    coopBi,
    confBiApprox,
    confChangeBiApprox,
    fluctBiApprox,
    fluctChangeBiApprox,
    energyBi,
    energyAlongConfChangeBi,
    energyAlongConfChangeNegBi,

    # from unfolding.jl
    freeEnergyOD,
    coopOD,
    coopNegOD,
    fluctOD,
    fluctChangeOD,
    energyOD,
    energyAlongConfCoorOD,
    energyAlongConfCoorNegOD,

    # from mcmc.jl
    U_full,
    run_mcmc,
    run_mcmc_para,
    computeΦΨFromConfs,
    make_2d_FreeEnergy
    
end
