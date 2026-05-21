# ==========================================
# Fichier : passerelle.jl
# ==========================================
using Pkg
Pkg.activate(".") 

using SynapseElife, Random, PiecewiseDeterministicMarkovProcesses, Sundials

# On définit bien les noms des arguments ici !
function simuler_synapse_brute(param_gamma_ampa, param_gamma_nmda)
    data_protocol = dataProtocol("TigaretMellor16")
    k = 8
    pls = 1
    start = 0.5e3

    events_times, is_pre_or_post_event = firingPattern(
        start_time   = start,
        n_pos        = data_protocol[!,:n_pos][k],
        delay_pos    = data_protocol[!,:delay_pos][k],
        n_pre        = data_protocol[!,:n_pre][k],
        delay_pre    = data_protocol[!,:delay_pre][k],
        delay        = data_protocol[!,:delay][k],
        pulse        = pls,
        freq         = data_protocol[!,:freq][k],
        causal       = data_protocol[!,:causal][k],
        repeat_times = data_protocol[!,:repeat_times][k],
        repeat_after = data_protocol[!,:repeat_after][k]
    )

    # Création du modèle avec les paramètres de l'IA
    param_synapse = SynapseParams(
        t_end         = start + 500.0,
        soma_dist     = 200.0,
        temp_rates    = data_protocol[!,:temp][k],
        Ca_ext        = data_protocol[!,:exca][k],
        Mg            = data_protocol[!,:exmg][k],
        age           = data_protocol[!,:age][k],
        injbap        = data_protocol[!,:inj_time][k],
        I_clamp       = data_protocol[!,:injection][k],
        sampling_rate = 10.0,
        
        # Injection : on utilise les mêmes noms que les arguments en haut
        gamma_ampa1   = param_gamma_ampa,
        gamma_ampa2   = param_gamma_ampa,
        gamma_ampa3   = param_gamma_ampa,
        gamma_nmda    = param_gamma_nmda
    )

    xc0 = initial_conditions_continuous_temp(param_synapse)
    xd0 = initial_conditions_discrete(param_synapse)

    is_glu_release, _, _, _, _, bap_by_epsp_times = stp(
        param_synapse.t_end, PreSynapseParams(h=0),
        events_times, is_pre_or_post_event, algo=CHV(CVODE_BDF())
    )

    result = evolveSynapse(
        xc0, xd0, param_synapse,
        events_times[events_times .< param_synapse.t_end],
        is_pre_or_post_event,
        Float64[],
        is_glu_release,
        (CHV(CVODE_BDF(linear_solver=:GMRES)), CHV(CVODE_BDF(linear_solver=:GMRES)));
        abstol = 1e-6, reltol = 1e-5, save_positions = (false, true), verbose = false
    )

    out = SynapseElife.get_names(result.XC, result.XD)
    
    return Vector(result.t), Vector(out[:Vsp])
end