using Test

@testset verbose = true "DifferentiationInterfaceJAX" begin
    @testset "Gradient" begin
        include("gradient.jl")
    end
end
