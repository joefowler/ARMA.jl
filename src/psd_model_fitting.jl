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

