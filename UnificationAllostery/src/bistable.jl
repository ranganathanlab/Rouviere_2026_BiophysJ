# --------------------------------------------------------------------
# Bistable quadratic potentials: helpers for energy, cooperativity, etc.
# --------------------------------------------------------------------

function freeEnergyBi(H::Matrix, h::Vector, g::Vector, β::Number)
    # Exact partition function for a quadratic double-well with offset g.
    @assert g[2:end] == zeros(length(g)-1) && g[1] >= 0
    N = length(g)
    H11 = H[1,1]
    Hbar = H[2:end,2:end] 
    t = H[2:end, 1]
    p = sqrt( (2*pi/β)^(N-1) / det(Hbar) )
    k1 = 0.5 * (H11 - t' * pinv(Hbar) * t)

    xa = norm(g)>sqrt(eps()) ? -g'*pinv(H)*(h-g) / norm(g) : 0
    ea = - 0.5 * (h-g)' * pinv(H) * (h-g)
    xb = norm(g)>sqrt(eps()) ? -g'*pinv(H)*(h+g) / norm(g) : 0
    eb = - 0.5 * (h+g)' * pinv(H) * (h+g)
    de =  eb - ea
    
    ta1 = 0.5 * p 
    ta2 = sqrt(pi /(β * k1))
    ta3 = erf(sqrt(β*k1) * xa) + 1
    Ia = ta1 * ta2 * ta3
    tb1 = 0.5 * p * exp(-β*de)
    tb2 = sqrt(pi /(β * k1))
    tb3 = 1 - erf(sqrt(β*k1) * xb) 
    Ib = tb1 * tb2 * tb3
 
    return ea + -log(Ia + Ib)/β
end

function coopBi(H::Matrix, h::Vector, g::Vector, ca::Vector, cb::Vector,
                Wa::Matrix, Wb::Matrix, β::Number)
    # ΔΔF computed from the exact bistable partition function.
    F00 = freeEnergyBi(H, h, g, β)
    F10 = freeEnergyBi(H + Wa, h + ca, g, β)
    F01 = freeEnergyBi(H + Wb, h + cb, g, β)
    F11 = freeEnergyBi(H + Wa + Wb, h + ca + cb, g, β)   
    return F10 + F01 - F00 - F11
end

function coopBi(h::Vector, ua::Vector, ub::Vector, λ::Number, β::Number,
                fa::Number, fb::Number, gmag::Number)
    N = length(ua)
    H = 10 * Matrix{Float64}(I, N,N)
    H[2,2] = λ
    normalize!(ua)
    normalize!(ub)
    ca = fa*ua
    cb = fb*ub
    Wa = 0*ua*ua'
    Wb = 0*ub*ub'
    g = zeros(N)
    g[1] = gmag
    #h = - (ca + cb) ./ 2
    return coopBi(H, h, g, ca, cb, Wa, Wb, β)
end

#################################################################

function confBiApprox(H::Matrix, h::Vector, g::Vector, β::Number)
    # Two-state approximation of ⟨r⟩ mixing the two basins.
    ea = - 0.5 * (h-g)' * pinv(H) * (h-g)
    eb = - 0.5 * (h+g)' * pinv(H) * (h+g)
    Z = exp(-β*ea) + exp(-β*eb)
    pa = exp(-β*ea) / Z
    pb = exp(-β*eb) / Z
    Ra = pinv(H) * (h-g)
    Rb = pinv(H) * (h+g)
    return pa * Ra + pb * Rb
end

function confChangeBiApprox(H::Matrix, h::Vector, g::Vector,
                             ca::Vector, cb::Vector, Wa::Matrix, Wb::Matrix, β)
    # Mean shift between apo and AB states using the two-state model.
    R00 = confBiApprox(H, h, g, β)
    R11 = confBiApprox(H + Wa + Wb, h + ca + cb, g, β)   
    return norm(R11 .- R00)
end

function confChangeBiApprox(h::Vector, ua::Vector, ub::Vector, λ::Number, β::Number,
                            fa::Number, fb::Number, gmag::Number)
    N = length(ua)
    H = 10 * Matrix{Float64}(I, N,N)
    H[2,2] = λ
    normalize!(ua)
    normalize!(ub)
    ca = fa*ua
    cb = fb*ub
    Wa = 0*ua*ua'
    Wb = 0*ub*ub'
    g = zeros(N)
    g[1] = gmag
    #h = - (ca + cb) / 2
    return confChangeBiApprox(H, h, g, ca, cb, Wa, Wb, β)
end


##################################################################

function fluctBiApprox(H, h, g, β)
    # Two-state estimate of √(<r^2> - <r>^2).
 
    ea = - 0.5 * (h-g)' * pinv(H) * (h-g)
    eb = - 0.5 * (h+g)' * pinv(H) * (h+g)
    Z = exp(-β*ea) + exp(-β*eb)
    pa = exp(-β*ea) / Z
    pb = exp(-β*eb) / Z
    Ra = pinv(H) * (h-g)
    Rb = pinv(H) * (h+g)
    R_mean = pa * Ra + pb * Rb 
    R2a = Ra' * Ra + tr(pinv(H)) / β
    R2b = Rb' * Rb + tr(pinv(H)) / β
    R2_mean = pa * R2a + pb * R2b
    return sqrt(R2_mean - R_mean' * R_mean)
end

function fluctChangeBiApprox(H::Matrix, h::Vector, g::Vector,
           ca::Vector, cb::Vector, Wa::Matrix, Wb::Matrix, β::Number)
    # |Δ fluctuation| between apo and AB states.
    return abs(fluctBiApprox(H+Wa+Wb, h+ca+cb, g, β) - fluctBiApprox(H,h,g,β))
    #return computeFluctBiApprox(H+Wa+Wb, h+fa+fb, g, β) - computeFluctBiApprox(H,h,g,β)
end

function fluctChangeBiApprox(h::Vector, ua::Vector, ub::Vector, λ::Number, β::Number,
                             fa::Number, fb::Number, gmag::Number)
    N = length(ua)
    H = 10 * Matrix{Float64}(I, N,N)
    H[2,2] = λ
    normalize!(ua)
    normalize!(ub)
    ca = fa*ua
    cb = fb*ub
    Wa = 0*ua*ua'
    Wb = 0*ub*ub'
    g = zeros(N)
    g[1] = gmag
    #h = - (ca + cb) / 2
    return fluctChangeBiApprox(H, h, g, ca, cb, Wa, Wb, β)
end


## Energy Landscape along Conformational Coor. ############################

function energyBi(r::Vector, H::Matrix, h::Vector, g::Vector)
    # Potential energy along a configuration r in the bistable model.
    return 0.5 * r'H*r - abs(g'r) - h'r
end

function energyAlongConfChangeBi(h::Vector, ua::Vector, ub::Vector, λ::Number, β::Number, 
                                 fa::Number, fb::Number, gmag::Number)
    # plot the energy along the conformational change
    # between unbound and doubly bound conditions.

    H = 10 * Matrix{Float64}(I, 5,5)
    H[2,2] = λ
    normalize!(ua)
    normalize!(ub)
    ca = fa*ua
    cb = fb*ub
    g = zeros(length(ua))
    g[1] = gmag

    # return the energy along the confo
    R00 =  confQuad(H, h-g) 
    R11 = confQuad(H, h + g + ca + cb) 
    ΔR = R11 - R00
    y = LinRange(-1, 2, 100)
    U00 = map(x -> energyBi(R00 + x*ΔR, H, h, g), y)
    U10 = map(x -> energyBi(R00 + x*ΔR, H, h+ca, g), y)
    U01 = map(x -> energyBi(R00 + x*ΔR, H, h+cb, g), y)
    U11 = map(x -> energyBi(R00 + x*ΔR, H, h+ca+cb, g), y)
    return U00, U10, U01, U11 
end

function energyAlongConfChangeNegBi(h::Vector, ua::Vector, ub::Vector, λ::Number, β::Number, 
                                 fa::Number, fb::Number, gmag::Number)
    # plot the energy along the conformational change for the negative cooperativity case
    # between unbound and doubly bound conditions.

    H = 10 * Matrix{Float64}(I, 5,5)
    H[2,2] = λ 
    normalize!(ua)
    normalize!(ub)
    ca = fa*ua
    cb = fb*ub
    g = zeros(length(ua))
    g[1] = gmag

    # return the energy along the confo
    R10 =  confQuad(H, h + g + ca) 
    R01 = confQuad(H, h - g + cb) 
    R00 = zeros(length(ua))
    ΔR = R01 - R10
    y = LinRange(-1, 1, 100)
    U00 = map(x -> energyBi(R00 + x*ΔR, H, h, g), y)
    U10 = map(x -> energyBi(R00 + x*ΔR, H, h+ca, g), y)
    U01 = map(x -> energyBi(R00 + x*ΔR, H, h+cb, g), y)
    U11 = map(x -> energyBi(R00 + x*ΔR, H, h+ca+cb, g), y)
    return U00, U10, U01, U11 
end


