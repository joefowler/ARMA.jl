include("RCPRoots.jl")
include("BarycentricRational.jl")
include("weightedAAA.jl")
include("PartialFracRational.jl")
include("vector_fitting.jl")

"""
    partial_frac(num, poles::Vector)

Compute the partial fraction decomposition of num / Π_i=1:n (z-α[i])
as ∑_i=1:n r[i]/(z-α[i]). Return the coefficient vector `r`.

Generally P(z)/Q(z) = ∑ P(α[i])/Q'(α[i]) * 1/(z-α[i])
"""
function partial_frac(num::Number, poles::AbstractVector)
    n = length(poles)
    T = promote_type(typeof(num), eltype(poles), Float64)
    r = zeros(T, n) .+ num

    for i=1:n
        for j=1:n
            j == i && continue
            r[i] /= (poles[i]-poles[j])
        end
    end
    r
end

function find_roots(vfit::PartialFracRational; steptol=1e-13)
    # The vfit is a model of F(z)=N(z)/D(z)+R(z)
    # Now find a model for F(z)/R(z) = N(z)/ [D(z)R(z)] + 1 as a partial fraction
    # Once it's a full partial fraction, it will have new poles (the roots of R) but
    # the same roots.
    p, q = vfit.n, vfit.m
    rough_roots = nothing
    if q<p
        rough_roots = roots_pfrac0(vfit.a, vfit.λ)
    elseif q==p
        rough_roots = roots_pfrac(vfit.a, vfit.λ, -vfit.b[end])
    else
        α = vfit.b[end]
        a = vfit.a / α
        bc = zeros(ComplexF64, q)
        β = ComplexF64[]
        β = legendre_roots(real(vfit.b))
        for i=1:p
            poles = [vfit.λ[i], β...]
            pfc = partial_frac(a[i], poles)
            bc[i] = pfc[1]
            bc[end-(q-p)+1:end] .+= pfc[2:end]
        end
        # rough_roots = eigvals(Diagonal(vcat(vfit.λ, β)) - bc*ones(q)')
        rough_roots = roots_pfrac1(-bc, vcat(vfit.λ, β))
    end

    rough_roots = RCPRoots(rough_roots)

    # Poor conditioning can make the eigenvalue method yield imperfect values. "Polish" those results
    # by a few Newton-Raphson steps
    function F(x)
        f0 = 0.0+0im
        for (ρ, λ) in zip(vfit.a, vfit.λ)
            f0 += ρ/(x-λ)
        end
        for (i, b) in zip(0:q-p, vfit.b)
            f0 += b*legendre(x, i)
        end
        f0
    end
    function dFdx(x)
        f0 = 0.0+0im
        for (ρ, λ) in zip(vfit.a, vfit.λ)
            f0 -= ρ/(x-λ)^2
        end
        for (i, b) in zip(1:q-p, vfit.b[2:end])  # skip the Legendre0, b/c its deriv = 0.
            f0 += b*dlegendre(x, i)
        end
        f0
    end

    r = copy(rough_roots.z)
    for i=1:length(r)
        # Try Newton steps, up to 5.
        x = r[i]
        for iter=1:5
            # Careful: very small _intended_ steps can actually yield no change at all
            # to floating-point precision. Check the actual step, not the intended one.
            intended_step = -F(x)/dFdx(x)
            xnew = x+intended_step
            actual_step = abs(xnew-x)
            x = xnew
            abs(actual_step) < steptol && break
        end
        r[i] = x
    end
    RCPRoots(r)
end
