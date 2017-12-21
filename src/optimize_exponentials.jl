using NLopt

type ExpFitBuffer
    p::Int
    Nt::Int
    Neval::Int

    r::Vector{Float64}
    t::Vector{Float64}
    w::Vector{Float64}
    wt::Vector{Float64}
    wrt::Vector{Float64}

    A::Vector{Complex128}
    B::Vector{Complex128}
    C::Vector{Float64}
    E::Vector{Complex128}
    G::Matrix{Complex128}
    M::Matrix{Complex128}
    N::Matrix{Complex128}
    Q::Matrix{Complex128}
    H::AbstractMatrix{Complex128}
    Bt::Matrix{Complex128}
    Θ::Matrix{Complex128}

    function ExpFitBuffer(r::Vector, t::AbstractVector, p::Integer, w::Vector)
        const Nt = length(r)
        @assert Nt == length(t)

        r = copy(r)
        t = copy(t)
        wt = w.*t
        wrt = wt.*r

        A = zeros(Complex128, p)
        C = zeros(Float64, p)
        B = zeros(Complex128, p)
        E = zeros(Complex128, p)
        G = zeros(Complex128, p, p)
        M = zeros(Complex128, p, p)
        N = zeros(Complex128, p, p)
        Q = zeros(Complex128, p, p)
        if p == 1
            H = zeros(Complex128, p, p)
        else
            H = Tridiagonal(M)
        end
        Bt = zeros(Complex128, Nt, p)
        Θ = zeros(Complex128, Nt, p)

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

    buffer.M = buffer.Bt'*(buffer.w.*buffer.Bt)
    buffer.N = buffer.Bt'*(buffer.wt.*buffer.Bt)
    for i=1:p; buffer.N[i,:] /= B[i]; end

    for i=1:p
        buffer.E[i] = dot(buffer.wt, buffer.Bt[:,i])/B[i]
    end

    for i=1:p
        buffer.Q[i,:] = A[i] * buffer.N[i,:]
    end

    buffer.G = pinv(buffer.M) * (diagm(buffer.E-buffer.N*A)-buffer.Q)

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
    const p = length(C)
    @assert p == buffer.p
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
    const p = length(guessC)
    if p == 0
        return sum(data[2:end].^2 .* w[2:end]), Float64[]
    end

    opt = Opt(:LD_MMA, p)
    const N = length(data)
    xtol_rel!(opt, 1e-6)
    ftol_rel!(opt, 1e-5)
    maxeval!(opt, 150p)

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

    t = 0:length(data)-1
    buffer = ExpFitBuffer(data, t, p, w)
    minf = ARMA_objective(guessC, [], buffer)
    # println("Before opt: $minf at $guessC")
    min_objective!(opt, (x,grad)->ARMA_objective(x, grad, buffer))
    (minf,minx,ret) = optimize(opt, guessC)
    # println("After opt: $minf at $minx after $(buffer.Neval) iterations (returned $ret)")
    minf, minx
end
