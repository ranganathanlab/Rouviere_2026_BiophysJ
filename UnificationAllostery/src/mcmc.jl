# Markov Chain Monte Carlo with Metropolis Hastings algorithm.

function U_full(r::AbstractVector,
                H::AbstractMatrix,
                H0::AbstractMatrix,
                g::AbstractVector,
                h::AbstractVector,
                ϵ::Number,
                β::Number)
    # The full energy of the model with a switch and unfolding.
    U_folded = 0.5*dot(r,H,r) - dot(h,r) - abs(dot(g,r))
    U_unfolded = 0.5*dot(r,H0,r) + ϵ
    return -log(exp(-β*U_folded) + exp(-β*U_unfolded))/β
end


function run_mcmc(n_iter::Int,
                  γ::Number,
                  β_mcmc::Number,
                  t_sample::Int,
                  r0::AbstractVector,
                  H::AbstractMatrix,
                  H0::AbstractMatrix,
                  g::AbstractVector,
                  h::AbstractVector,
                  ϵ::Number,
                  β::Number)

    # Simple random-walk Metropolis Hastings sampler.
    l = length(r0)
    r = copy(r0) # detach from outside function
    r_ = similar(r0)
    E = U_full(r, H, H0, g, h, ϵ, β)

    confs = Matrix{Float64}(undef, l, Int(n_iter/t_sample))
    n_accepted = 0

    for t in 1:n_iter # MCMC loop
        # generate new step
        for i in 1:l
            r_[i] = r[i] + γ*randn()
        end
        E_ = U_full(r_, H, H0, g, h, ϵ, β) # Compute new energy
        if rand() < exp(-β_mcmc*(E_ - E)) # Metropolis condition.
            E = E_
            r .= r_
            n_accepted += 1
        end
        
        if t % t_sample == 0 # save current state
            i = Int(t / t_sample)
            confs[:,i] = r
        end
    end
    println("Fraction accepted: ", n_accepted / n_iter)
    return confs

end

function computeΦΨFromConfs(confs::AbstractMatrix, ΔR::AbstractVector, R00::AbstractVector)
    # Compute collective variables Φ (projection) and Ψ (orthogonal rms) per sample.
    ΔR_hat = normalize(ΔR)
    r_tmp = zeros(size(confs,1))
    r_x = similar(r_tmp)
    Φ_vec = zeros(size(confs,2))
    Ψ_vec = similar(Φ_vec)

    for i in eachindex(Φ_vec)
        @views r = confs[:,i]
        @. r_tmp = r - R00
        d = ΔR_hat ⋅ r_tmp 
        @. r_x = r_tmp - d * ΔR_hat 
        Φ_vec[i] = d
        Ψ_vec[i] = norm(r_x)
    end
    return Φ_vec, Ψ_vec
end

function make_2d_FreeEnergy(x, y, β, x_edges, y_edges)
    # Estimate the 2-D free-energy landscape from sampled CVs.
    data = (x,y)
    edges = (x_edges, y_edges)
    h = fit(Histogram, data, edges)
    P = Matrix{Float64}(h.weights)'
    P ./= sum(P)
    for i in eachindex(P)
        P[i] == 0 && (P[i] = NaN)
    end
    return - log.(P) ./ β # free energy
end

