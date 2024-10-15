using DifferentiationInterface
using DifferentiationInterfacePython
using Test

@testset verbose = true "DifferentiationInterfacePython" begin
    @testset "Basic gradient" begin
        backend = AutoJAX()
        f(x) = sum(x)
        @test gradient(f, backend, [3.0]) == [1.0]
    end
end
