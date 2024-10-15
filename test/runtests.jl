using Test

@testset verbose = true "DifferentiationInterfaceJAX" begin
    @testset "Gradient" begin
        include("gradient.jl")
    end
    @testset "Pushforward" begin
        include("pushforward.jl")
    end
    @testset "Jacobian" begin
        include("jacobian.jl")
    end
end
