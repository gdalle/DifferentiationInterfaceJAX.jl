using DifferentiationInterface
using DifferentiationInterfacePython
using Test

@testset verbose = true "DifferentiationInterfacePython" begin
    @testset "Basic gradient" begin
        backend = AutoAutoGrad()
        f(x) = sum(x)
        prepare_gradient(f, backend, [3.0])
        @test gradient(f, backend, [3.0]) == [1.0]
    end
end
