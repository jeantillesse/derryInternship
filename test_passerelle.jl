using Test
using Pkg

# S'assurer que l'environnement courant est actif
Pkg.activate(".")

# Inclure le fichier à tester
include("passerelle.jl")

@testset "Tests passerelle.jl" begin
    @testset "simulateur_sbi" begin
        # Vecteur de paramètres par défaut : [N_ampa, N_nmda, N_ca*, L_neck]
        params = [120.0, 15.0, 3.0, 0.2]
        
        # Exécution avec la stratégie parallèle
        resultats = simulateur_sbi(ParallelStrategy(), [params])
        
        # Vérifications
        @test resultats isa Vector{Vector{Float64}}
        @test length(resultats) == 1
        
        result = resultats[1]
        @test length(result) == 14
        @test all(isfinite.(result))
    end
end
