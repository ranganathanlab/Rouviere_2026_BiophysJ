
# --------------------------------------------------------------------
# Two-state order-disorder model helpers (folded/unfolded quadratic wells).
# --------------------------------------------------------------------

function freeEnergyOD(H, H0, ϵ, β)
    # Free energy combining folded and unfolded wells (no external force).
    zerosvec = zeros(size(H,1))
    F_dis = freeEnergyQuad(H0, zerosvec, β) + ϵ
    F_ord= freeEnergyQuad(H + H0, zerosvec, β)
    return -log(exp(-β*F_dis) + exp(-β*F_ord)) / β
end

function coopOD(H, H0, Wa, Wb, ϵ, β)
    # ΔΔF when both ligands stabilize the folded state.
    F00 = freeEnergyOD(H, H0, ϵ, β)
    F10 = freeEnergyOD(H + Wa, H0 + Wa, ϵ, β)
    F01 = freeEnergyOD(H + Wb, H0 + Wb, ϵ, β)
    F11 = freeEnergyOD(H + Wa+Wb, H0 + Wa + Wb, ϵ, β)
    return F10 + F01 - F00 - F11
end


function coopOD(ua::Vector, ub::Vector, λ::Number, β::Number,
                ka::Number, kb::Number, ϵ::Number; stiffnessDisordered=1e-2)
    N = length(ua)
    H = 10 * Matrix{Float64}(I, N,N)
    H[2,2] = λ
 
    H0 = stiffnessDisordered * Matrix{Float64}(I, N,N)
    normalize!(ua)
    normalize!(ub)
    Wa = ka*ua*ua'
    Wb = kb*ub*ub'
    return coopOD(H, H0, Wa, Wb, ϵ, β)
end


function coopNegOD(ua::Vector, ub::Vector, λ::Number, β::Number,
                ka::Number, kb::Number, ϵ::Number; stiffnessDisordered=1e-2)
    N = length(ua)
    H = 10 * Matrix{Float64}(I, N,N)
    H[2,2] = λ
 
    H0 = stiffnessDisordered * Matrix{Float64}(I, N,N)
    normalize!(ua)
    normalize!(ub)
    Wa = ka*ua*ua'
    Wb = kb*ub*ub'

    H -= 1.01*Wb # added keep the curvature of the basins postive upon binding.
    H0 -= 1.01*Wb

    return coopOD(H, H0, Wa, Wb, ϵ, β)
end


##################

function fluctOD(H, H0, ϵ, β)
    # Two-state RMS fluctuation combining folded/unfolded basins.
 
    b_dis = exp(-β*ϵ) * det(H0)^(-0.5)
    b_ord = det(H)^(-0.5)
    Z = b_ord + b_dis
    p_dis = b_dis / Z
    p_ord = b_ord / Z

    R2_dis = tr(pinv(H0)) / β
    R2_ord = tr(pinv(H + H0)) / β
    R2_mean = p_dis * R2_dis + p_ord * R2_ord
    return sqrt(R2_mean)
end

function fluctChangeOD(H, H0, Wa, Wb, ϵ, β)
    # Absolute change in fluctuations upon binding both ligands.
    fl00 = fluctOD(H, H0, ϵ, β) 
    fl11 = fluctOD(H+Wa+Wb, H0+Wa+Wb, ϵ, β) 
    return abs(fl11 - fl00)
end


function fluctChangeOD(ua::Vector, ub::Vector, λ::Number, β::Number,
                         ka::Number, kb::Number, ϵ::Number; stiffnessDisordered=1e-2)
    N = length(ua)
    H = 10 * Matrix{Float64}(I, N,N)
    H[2,2] = λ 
    H0 = stiffnessDisordered * Matrix{Float64}(I, N,N)
    normalize!(ua)
    normalize!(ub)
    Wa = ka*ua*ua'
    Wb = kb*ub*ub'
    return fluctChangeOD(H, H0, Wa, Wb, ϵ, β) 
end

###########################################


function energyOD(r::Vector, H::Matrix, H0::Matrix, β::Number, ϵ::Number)
    # Effective free energy along coordinate r including folding penalty.
    U_ord = 0.5*r'H*r
    U_dis = 0.5*r'H0*r + ϵ 
    return -log(exp(-β*U_ord) + exp(-β*U_dis))/β 
end


function energyAlongConfCoorOD(ua::Vector, ub::Vector, λ::Number,
             β::Number, ka::Number, kb::Number, ϵ::Number; stiffnessDisordered=1e-2)
    N = length(ua)
    H = 10 * Matrix{Float64}(I, N,N)
    H[2,2] = λ
    H0 = stiffnessDisordered * Matrix{Float64}(I, N,N)
    normalize!(ua)
    normalize!(ub)
    Wa = ka*ua*ua'
    Wb = kb*ub*ub'
    
    # Energy profile along the soft coordinate (2nd axis).
    R00 = zeros(N)
    ΔR =  zeros(N)
    ΔR[2] = 1 # use the soft mode as the conf coor
    
    y = LinRange(-5,5, 100)
    U00 = map(x -> energyOD(R00 + x*ΔR, H, H0, β, ϵ), y)
    U10 = map(x -> energyOD(R00 + x*ΔR, H+Wa, H0+Wa, β, ϵ), y)
    U01 = map(x -> energyOD(R00 + x*ΔR, H+Wb, H0+Wb, β, ϵ), y)
    U11 = map(x -> energyOD(R00 + x*ΔR, H+Wa+Wb, H0+Wa+Wb, β, ϵ), y)
    return U00, U10, U01, U11
end


function energyAlongConfCoorNegOD(ua::Vector, ub::Vector, λ::Number,
             β::Number, ka::Number, kb::Number, ϵ::Number; stiffnessDisordered=1e-2)
    N = length(ua)
    H = 10 * Matrix{Float64}(I, N,N)
    H[2,2] = λ 
    H0 = stiffnessDisordered * Matrix{Float64}(I, N,N)
    normalize!(ua)
    normalize!(ub)
    Wa = ka*ua*ua'
    Wb = kb*ub*ub'
    
    # Energy profile along the soft coordinate (2nd axis).
    R00 = zeros(N)
    ΔR =  zeros(N)
    ΔR[2] = 1 # use the soft mode as the conf coor
    
    H -= 1.01*Wb # added keep the curvature of the basins postive upon binding.
    H0 -= 1.01*Wb
    
    y = LinRange(-5,5, 100)
    U00 = map(x -> energyOD(R00 + x*ΔR, H, H0, β, ϵ), y)
    U10 = map(x -> energyOD(R00 + x*ΔR, H+Wa, H0+Wa, β, ϵ), y)
    U01 = map(x -> energyOD(R00 + x*ΔR, H+Wb, H0+Wb, β, ϵ), y)
    U11 = map(x -> energyOD(R00 + x*ΔR, H+Wa+Wb, H0+Wa+Wb, β, ϵ), y)
    return U00, U10, U01, U11
end
