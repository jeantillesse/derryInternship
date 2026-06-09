using Pkg
Pkg.activate(".")

include("passerelle.jl")

params = [120.0, 15.0, 3.0, 0.2]

println("--- Échauffement ---")
# Échauffement pour forcer la compilation de Julia (sinon le premier test est toujours pénalisé)
simulateur_complet_sbi(params, multithread=false)
simulateur_complet_sbi(params, multithread=true)

println("\n--- Test Single-thread ---")
temps_single = @elapsed begin
    simulateur_complet_sbi(params, multithread=false)
end
println("Temps d'exécution sans multi-threading : ", round(temps_single, digits=2), " secondes")

println("\n--- Test Multi-thread ---")
temps_multi = @elapsed begin
    simulateur_complet_sbi(params, multithread=true)
end
println("Temps d'exécution avec multi-threading : ", round(temps_multi, digits=2), " secondes")

gain = (temps_single - temps_multi) / temps_single * 100
acc = temps_single / temps_multi
println("\nAccélération : x", round(acc, digits=2), " (Gain de temps : ", round(gain, digits=2), "%)")
