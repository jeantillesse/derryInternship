using Test
using Pkg

# S'assurer que l'environnement courant est actif pour avoir accès aux dépendances
Pkg.activate(".")

# Inclure le fichier à tester
include("passerelle.jl")

@testset "Tests passerelle.jl" begin
    @testset "simulateur_complet_sbi" begin
        # Vecteur de paramètres par défaut : [N_ampa, N_nmda, N_ca*, L_neck]
        params = [120.0, 15.0, 3.0, 0.2]
        
        # Exécution de la simulation. 
        # Remarque : cela peut prendre un certain temps selon la puissance de calcul
        # car on simule N_SWEEPS pour chaque protocole.
        result = simulateur_complet_sbi(params)
        
        # Vérifications
        @test result isa Vector{Float64}
        # On attend 7 protocoles x 2 statistiques (moyenne, variance) = 14 valeurs
        @test length(result) == 14
        @test all(isfinite.(result))
    end
end
