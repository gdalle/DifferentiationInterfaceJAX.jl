using Aqua
using JET
using JuliaFormatter
using DifferentiationInterfaceJAX
using Test

@testset "Aqua" begin
    Aqua.test_all(DifferentiationInterfaceJAX)
end;

@testset "JET" begin
    JET.test_package(DifferentiationInterfaceJAX; target_defined_modules=true)
end;

@testset "JuliaFormatter" begin
    @test JuliaFormatter.format(DifferentiationInterfaceJAX; overwrite=false)
end;
