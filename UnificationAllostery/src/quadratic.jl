# --------------------------------------------------------------------
# Quadratic-model utilities shared by several figure scripts.
# --------------------------------------------------------------------

## Cooperativity ############################################
function freeEnergyQuad(H::Matrix, h::Vector, β::Number)
    # Free energy for V(r) = 0.5 r' H r - h' r with invertible H.
    N = size(H,1)
    E = - 0.5 * h' * inv(H) * h # ground state energy
    Fhar = -log( (2 * pi / β)^N / det(H) ) / (2β)
    return E + Fhar
end

function coopQuad(H::Matrix, h::Vector, ca::Vector, cb::Vector,
                 Wa::Matrix, Wb::Matrix, β::Number)
    # Compute ΔΔF for a general quadratic potential.
    F00 = freeEnergyQuad(H, h, β)
    F10 = freeEnergyQuad(H + Wa, h + ca, β)
    F01 = freeEnergyQuad(H + Wb, h + cb, β)
    F11 = freeEnergyQuad(H + Wa + Wb, h + ca + cb, β)   
    return F10 + F01 - F00 - F11
end

function coopQuad(ua::Vector, ub::Vector, λ::Number, β::Number,
                  fa::Number, fb::Number, ka::Number, kb::Number)
    # Convenience wrapper that builds the Hessian from binding-direction vectors.
    N = length(ua)
    H = 10 * Matrix{Float64}(I, N,N)
    H[2,2] = λ
    normalize!(ua)
    normalize!(ub)
    ca = fa*ua
    cb = fb*ub
    Wa = ka*ua*ua'
    Wb = kb*ub*ub'
    h = zeros(N)
    return coopQuad(H, h, ca, cb, Wa, Wb, β)
end



## Conformation and Conf Change ##################################

confQuad(H::Matrix, h::Vector) = inv(H) * h

function confChangeQuad(H::Matrix, h::Vector, ca::Vector, cb::Vector, Wa::Matrix, Wb::Matrix)
    # Mean shift between the apo and doubly-liganded states.
    R00 = confQuad(H, h)
    R11 = confQuad(H + Wa + Wb, h + ca + cb)   
    return norm(R11 .- R00)
end

function confChangeQuad(ua::Vector, ub::Vector, λ::Number, β::Number,
                        fa::Number, fb::Number, ka::Number, kb::Number)
    # compute the cooperativity of system with soft mode with 
    # stiffness λ and overlaps with modes 1,2, of O1, O2.
    # soft mode is the second mode.
    N = length(ua)
    H = 10 * Matrix{Float64}(I, N,N)
    H[2,2] = λ
    normalize!(ua)
    normalize!(ub)
    ca = fa*ua
    cb = fb*ub
    Wa = ka*ua*ua'
    Wb = kb*ub*ub'
    h = zeros(N)
    return confChangeQuad(H, h, ca, cb, Wa, Wb)
end


## Fluctuation and Fluctuational change ######################################

function fluctQuad(H::Matrix, β::Number)
    # Compute √(<r^2> - <r>^2) for the quadratic distribution.
    return sqrt(tr(inv(H))) / β^2
end

function fluctChangeQuad(H::Matrix, Wa::Matrix, Wb::Matrix, β::Number)
    # Difference in fluctuation magnitude between apo and AB states.
    return fluctQuad(H + Wa + Wb, β) - fluctQuad(H, β)
end

function fluctChangeQuad(ua::Vector, ub::Vector, λ::Number, β::Number,
                         fa::Number, fb::Number, ka::Number, kb::Number)
    # compute the cooperativity of system with soft mode with 
    # stiffness λ and overlaps with modes 1,2, of O1, O2.
    # soft mode is the second mode.
    N = length(ua)
    H = 10 * Matrix{Float64}(I, N,N)
    H[2,2] = λ
    normalize!(ua)
    normalize!(ub)
    Wa = ka*ua*ua'
    Wb = kb*ub*ub'
    return fluctChangeQuad(H, Wa, Wb, β)
end
