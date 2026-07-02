using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using SynapseElife, Random, PiecewiseDeterministicMarkovProcesses, Sundials, Statistics

const DATA_PROTOCOL = dataProtocol("TigaretMellor16")

function simuler_synapse_courbe(val_ampa, val_nmda, val_ca, val_neck, k)
    # Safety boundaries
    val_neck = max(val_neck, 0.05) 
    val_ampa = max(1.0, val_ampa)
    val_nmda = max(1.0, val_nmda)
    val_ca   = max(1.0, val_ca)

    start = 0.5e3
    pls = DATA_PROTOCOL[!, :pulse][k]

    events_times, is_pre_or_post_event = firingPattern(
        start_time   = start,
        n_pos        = DATA_PROTOCOL[!,:n_pos][k],
        delay_pos    = DATA_PROTOCOL[!,:delay_pos][k],
        n_pre        = DATA_PROTOCOL[!,:n_pre][k],
        delay_pre    = DATA_PROTOCOL[!,:delay_pre][k],
        delay        = DATA_PROTOCOL[!,:delay][k],
        pulse        = pls,
        freq         = DATA_PROTOCOL[!,:freq][k],
        causal       = DATA_PROTOCOL[!,:causal][k],
        repeat_times = DATA_PROTOCOL[!,:repeat_times][k],
        repeat_after = DATA_PROTOCOL[!,:repeat_after][k]
    )

    afterend = 1.5e5
    param_synapse = SynapseParams(
        t_end         = start + (DATA_PROTOCOL[!,:repeat_times][k]+1)*pls*1000/DATA_PROTOCOL[!,:freq][k] + afterend,
        soma_dist     = 200.0,
        temp_rates    = DATA_PROTOCOL[!,:temp][k],
        Ca_ext        = DATA_PROTOCOL[!,:exca][k],
        Mg            = DATA_PROTOCOL[!,:exmg][k],
        age           = DATA_PROTOCOL[!,:age][k],
        injbap        = DATA_PROTOCOL[!,:inj_time][k],
        I_clamp       = DATA_PROTOCOL[!,:injection][k],
        sampling_rate = 0.1,  
        N_ampa        = round(Int, val_ampa),
        N_NMDA        = round(Int, val_nmda),
        N_caT         = round(Int, val_ca),
        N_caR         = round(Int, val_ca),
        N_caL         = round(Int, val_ca),
        L_neck        = val_neck
    )

    p = param_synapse.p_release
    pre_synapse = PreSynapseParams(h = (p[4]+ p[3]/(1 + exp(p[1] * (DATA_PROTOCOL[!,:exca][k] - p[2])))))

    xc0 = initial_conditions_continuous_temp(param_synapse)
    xd0 = initial_conditions_discrete(param_synapse)

    is_glu_release, _, _, _, _, bap_by_epsp_times = stp(
        param_synapse.t_end, pre_synapse,
        events_times, is_pre_or_post_event, _plot = false, algo = CHV(CVODE_BDF())
    )

    valid_indices = events_times .< param_synapse.t_end
    events_valides = events_times[valid_indices]
    pre_post_valides = is_pre_or_post_event[valid_indices]
    glu_valides = is_glu_release[valid_indices]

    result = evolveSynapse_noformat(
        xc0, xd0, param_synapse,
        events_valides,
        pre_post_valides, 
        DATA_PROTOCOL[!,:AP_by_EPSP][k] == "yes" ? bap_by_epsp_times : Float64[], 
        glu_valides,
        (CHV(CVODE_BDF(linear_solver=:GMRES)), CHV(CVODE_BDF(linear_solver=:GMRES)));
        abstol = 1e-6, reltol = 1e-5, save_positions = (false, true), verbose = false
    )

    # Return time (in seconds) and normalized weight change in percent
    # NC is index 36, LTD is index 37, LTP is index 38 in result.XD
    t_sec = result.t ./ 1000.0
    
    NC_curve = result.XD[36, :]
    LTD_curve = result.XD[37, :]
    LTP_curve = result.XD[38, :]
    

    N_total = NC_curve .+ LTD_curve .+ LTP_curve
    N_total = max.(1.0, N_total)
    
    delta_W = (NC_curve .* 1.0 .+ LTD_curve .* 0.3 .+ LTP_curve .* 3.5) ./ N_total
    
    # We return the actual weight change: (delta_W - 1.0)
    # The Python script will multiply by 100 to get percentages
    weight_change = delta_W .- 1.0

    return (t = t_sec, weight_change = weight_change)
end

function simuler_les_7_protocoles(val_ampa, val_nmda, val_ca, val_neck)
    res_dict = Dict{String, Any}()
    # Tigaret's 7 protocols are at indices 2 to 8 of TigaretMellor16
    for k in 2:8
        proto_name = DATA_PROTOCOL[k, :protocol]
        println("Simulating protocol: $proto_name (index $k)...")
        curve = simuler_synapse_courbe(val_ampa, val_nmda, val_ca, val_neck, k)
        res_dict[proto_name] = curve
    end
    return res_dict
end
