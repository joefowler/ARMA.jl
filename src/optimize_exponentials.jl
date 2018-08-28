using NLopt

mutable struct ExpFitBuffer
    p::Int
    Nt::Int
    Neval::Int

    r::Vector{Float64}
    t::Vector{Float64}
    w::Vector{Float64}
    wt::Vector{Float64}
    wrt::Vector{Float64}

    A::Vector{ComplexF64}
    B::Vector{ComplexF64}
    C::Vector{Float64}
    E::Vector{ComplexF64}
    G::Matrix{ComplexF64}
    M::Matrix{ComplexF64}
    N::Matrix{ComplexF64}
    Q::Matrix{ComplexF64}
    H::AbstractMatrix{ComplexF64}
    Bt::Matrix{ComplexF64}
    Θ::Matrix{ComplexF64}

    function ExpFitBuffer(r::Vector, t::AbstractVector, p::Integer, w::Vector)
        Nt = length(r)
        @assert Nt == length(t)

        r = copy(r)
        t = copy(t)
        wt = w.*t
        wrt = wt.*r

        A = zeros(ComplexF64, p)
        C = zeros(Float64, p)
        B = zeros(ComplexF64, p)
        E = zeros(ComplexF64, p)
        G = zeros(ComplexF64, p, p)
        M = zeros(ComplexF64, p, p)
        N = zeros(ComplexF64, p, p)
        Q = zeros(ComplexF64, p, p)
        if p == 1
            H = zeros(ComplexF64, p, p)
        else
            H = Tridiagonal(M)
        end
        Bt = zeros(ComplexF64, Nt, p)
        Θ = zeros(ComplexF64, Nt, p)

        new(p, Nt, 0, r, t, w, wt, wrt, A, B, C, E, G, M, N, Q, H, Bt, Θ)
    end
end



function ARMA_gradient{T<:Number}(grad::Vector, buffer::ExpFitBuffer, A::Vector{T}, B::Vector{T}, fmodel::Vector)
    t = buffer.t
    p = buffer.p

    for i = 1:p
        buffer.Bt[:,i] = B[i].^t
        buffer.Θ[:,i] = A[i]/B[i] * t .* buffer.Bt[:,i]
    end

    buffer.M[:,:] = buffer.Bt'*(buffer.w.*buffer.Bt)
    buffer.N[:,:] = buffer.Bt'*(buffer.wt.*buffer.Bt)
    for i=1:p; buffer.N[i,:] /= B[i]; end

    for i=1:p
        buffer.E[i] = dot(buffer.wt, buffer.Bt[:,i])/B[i]
    end

    for i=1:p
        for j=1:p
            buffer.Q[i,j] = A[i] * buffer.N[i,j]
        end
    end

    buffer.G[:,:] = pinv(buffer.M) * (diagm(buffer.E-buffer.N*A)-buffer.Q)

    H = buffer.H
    for i=1:2:p-1
        ID = 1./(B[i]-B[i+1])
        H[i,i] = -B[i]*ID
        H[i+1,i] = B[i+1]*ID
        H[i,i+1] = -ID
        H[i+1,i+1] = ID
    end
    if p%2 == 1
        H[end,end] = B[end]^2
    end

    Jacobian = real((buffer.Θ + buffer.Bt*buffer.G)*buffer.H)
    grad[:] = 2*(Jacobian' * (buffer.w.*(fmodel-buffer.r)))
end


function ARMA_objective(C::Vector, grad::Vector, buffer::ExpFitBuffer)
    p = length(C)
    B = C2B(C)
    A = findA(buffer.t, buffer.r, B, w=buffer.w)
    fmodel = exponential_model(buffer.t, A, B)

    if length(grad) > 0
        ARMA_gradient(grad, buffer, A, B, fmodel)
    end
    buffer.Neval += 1
    sum(buffer.w .* (fmodel - buffer.r).^2)
end


function optimize_exponentials(data::Vector, w::Vector, guessC::Vector)
    p = length(guessC)
    if p == 0
        return sum(data[2:end].^2 .* w[2:end]), Float64[]
    end

    opt = Opt(:LD_MMA, p)
    opt = Opt(:LN_BOBYQA, p)
    xtol_abs!(opt, 1e-6)
    ftol_rel!(opt, 1e-5)
    maxeval!(opt, 100p)

    # Constraint bounds on the quadratic representation of the bases.
    # As all bases are in the unit circle, the linear coefficients are in (-2,2)
    # and the quadratic ones are in (-1,1).
    lb = zeros(guessC)-2
    ub = zeros(guessC)+2
    lb[2:2:end] = -1
    ub[2:2:end] = +1
    if p%2 == 1
        lb[end] = -Inf
        ub[end] = +Inf
    end
    lower_bounds!(opt, lb)
    upper_bounds!(opt, ub)
    step = 0.1*(ub-lb)
    if step[end] > 1
        step[end] = guessC[end]*0.5
    end
    initial_step!(opt, step)

    t = 0:length(data)-1
    buffer = ExpFitBuffer(data, t, p, w)
    min_objective!(opt, (x,grad)->ARMA_objective(x, grad, buffer))
    minf1 = ARMA_objective(guessC, [], buffer)
    # println("Before opt: $minf1 at $guessC")
    try
        (minf,minx,ret) = optimize(opt, guessC)
        # println("After opt: $minf at $minx after $(buffer.Neval) iterations (returned $ret)")
        # println("Moved by $(minx-guessC), improved by $(minf1-minf)\n")
        return minf, minx
    catch e
        @printf("Error in %d-order fit: %s\n", p, e)
        return minf1, guessC
    end
end
