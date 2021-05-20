# Perform Vector Fitting, a fixed-point method for rational-function approximation.
#
# Vector fitting fits a low-order rational function to one or more (hence, "vector") data sets, where the
# multiple functions share common poles. We do not implement the multiple-function version here.
#
# Goal: match a data vector to the rational function model
#
# F(z) = N(z)/D(z) + R(z)
#
# where N, D, and R are the numerator, denominator, and remainder polynomials, of degree n-1, n, m-n
# respectively. Require m≥n-1. (In the case of m=n-1, we take the "degree -1 polynomial" to equal 0.)
# This problem is nonlinear because of the parameters of D(z) that appear in the denominator (in effect,
# the poles of F). If the poles of F were known, the problem would be linear.
#
# Vector fitting approaches this nonlinear problem by rewriting N(z) and D(z) in barycentric form. That
# is, we write N(z) as the product of the partial-fraction rational function n(z) with some set of n nodes
# called λ, times the Legendre-form polynomial of degree n with roots at these nodes λ.
# n(z) = ∑ i=1:n a[i]/(z-λ[i])
# N(z) = n(z) ∏ i=1:n (z-λ[i])
#
# Similarly, the denominator D(z) is the product of another rational function d(z). This d is a constant
# plus a partial-fraction form with the same p nodes λ.
# d(z) = 1 + ∑ i=1:n b[i]/(z-λ[i])
# D(z) = d(z) ∏ i=1:n (z-λ[i])
# Without loss of generality, we are assuming that the constant term in d(z) is 1. These definitions let
# N and D be general polynomials of degree n-1 and n (apart from the scale freedom we took from D).
#
# Now F(z) = N(z)/D(z) + R(z) = n(z)/d(z) + R(z)
#
# The idea of vector fitting is that if we can choose the nodes to be the poles of the target function F, then
# n(z) function will contain all the information, the d(z) denominator will be unity, and the nonlinear
# coefficients b will all be zero. In that limit, n(z) and R(z) will contain all the remaining unknowns, and
# they can be found in a linear problem. That is, fitting F to data samples s becomes equivalent to
# fitting F(Z)*d(z)=n(z)+d(z)*R(z) to s*d(z) in the limit that d → 1. We try to approach that limit
# iteratively.
#
# Based on Gustavsen, B., & Semlyen, A. (1999). "Rational approximation of frequency domain responses by
# vector fitting." _IEEE Transactions on Power Delivery_, 14(3), 1052–1061. https://doi.org/10.1109/61.772353

vectorfit(z::AbstractVector, f::Function, λ0::AbstractVector, m=-1; maxit=10) =
    vectorfit(z, f.(z), ones(eltype(z), length(z)), λ0, m; maxit)

vectorfit(z::AbstractVector, f::Function, wt::AbstractVector, λ0::AbstractVector, m=-1; maxit=10) =
    vectorfit(z, f.(z), wt, λ0, m; maxit)

vectorfit(z::AbstractVector, f::AbstractVector, λ0::AbstractVector, m=-1; maxit=10) =
    vectorfit(z, f, ones(eltype(z), length(z)), λ0, m; maxit)

"""
    vectorfit(z, f, [wt,] λguess, m=-1; maxit=10)

Use the vector fitting algorithm of Gustavsen & Semlyen 1999 https://doi.org/10.1109/61.772353 to make a
low-order rational function model for f(z) given samples of f.

# Arguments
- `z`: vector of sample locations
- `f`: vector of function values at `z` (must be same length as `z`) or a callable object that will be called for each `z`
- `wt`: vector of statistical weights to give to each sample. If omitted, it will default to equal for all samples.
- `λguess`: vector of length `n`, an initial guess about the location of the `n` poles; may be of type `RCPRoots`
  to restrict poles to be like roots of a real-coefficient polynomial (that is, pole `p` is allowed if and only if
  `conj(p)` appears, too). This guess determines the order of the rational function's denomerator.
- `m`: the order of the rational function's numerator. Defaults to `n`, but may be any integer ≥ `n-1`.
- `maxit`: the maximum number of vector fit iterations to perform.
"""
function vectorfit(z::AbstractVector, f::AbstractVector, wt::AbstractVector, λ0::AbstractVector, m::Integer=-1; maxit=10)
    n = length(λ0)
    if m < 0
        m = n  # Default to an (n,n) rational function fit
    end
    if m < n-1
        throw(ErrorException("vectorfit called with degree (n,m)=($n,$m). Requires m ≥ n-1."))
    end

    N = length(z)
    if N != length(f)
        throw(DimensionMismatch("vectorfit: length(z) $N != length(f) $(length(f))"))
    elseif N != length(wt)
        throw(DimensionMismatch("vectorfit: length(z) $N != length(wt) $(length(wt))"))
    end

    # Get types promoted. Final type: P
    z, f, wt, _ = promote(collect(z), f, wt, real(λ0))
    P = eltype(z)
    W = Diagonal(sqrt.(wt))
    Wf = W*f

    C = Array{Complex{P}}(undef, N, n)
    Φ = Array{Complex{P}}(undef, N, m-(n-1))

    # Use Legendre polynomials for the basis
    zmin, zmax = minimum(z), maximum(z)
    zscaled = 2(z.-zmin)/(zmax-zmin) .- 1
    for k=1:m-(n-1)
        Φ[:,k] .= legendre.(zscaled, k-1)
    end
    b = nothing
    model = nothing
    λ = λ0
    @show λ0

    for iter=1:maxit
        for k=1:n
            C[:,k] .= 1.0 ./ (z.-λ[k])
        end

        D = Diagonal(f)*C
        M = W*hcat(C, Φ, D)
        optparam = M\Wf
        # ρ = optparam[1:n]
        # c = optparam[n+1:m+1]
        b = optparam[m+2:end]

        # Update the poles to those implied by the new d(z) function
        λ = roots_pfrac1(b, λ)
    end
    @show λ

    for k=1:n
        C[:,k] .= 1.0 ./ (z.-λ[k])
    end
    M = W*hcat(C, Φ)
    optparam = M\Wf
    model = real(hcat(C, Φ)*optparam)

    Wtres = norm(W*model.-Wf)
    ρ = optparam[1:n]
    c = real(optparam[n+1:end])
    PartialFracRational(λ, ρ, c; polyMin=zmin, polyMax=zmax)
end
