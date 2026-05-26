using Pkg
Pkg.activate(".") 

using SynapseElife, Random, PiecewiseDeterministicMarkovProcesses, Sundials

function simuler_synapse_brute(val_n_ampa, val_n_nmda, val_n_vgcc, val_l_neck, k_protocole)
    data_protocol = dataProtocol("TigaretMellor16")
    k = k_protocole
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
        
        N_ampa = round(Int, val_n_ampa),
        N_NMDA = round(Int, val_n_nmda),
        N_caT  = round(Int, val_n_vgcc),
        N_caR  = round(Int, val_n_vgcc),
        N_caL  = round(Int, val_n_vgcc),
        L_neck = val_l_neck
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
    # 1. On récupère la toute dernière valeur (à la fin du protocole) de act_P et act_D
    act_D_final = result.XC[27, end]
    act_P_final = result.XC[28, end]

    # 2. Formule de test (Readout fictif)
    delta_W = 1.0 + (0.05 * act_P_final) - (0.02 * act_D_final)

    # 3. Sécurité biologique
    delta_W = max(0.1, delta_W)

    return delta_W
end