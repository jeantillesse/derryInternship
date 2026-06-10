using Pkg
Pkg.activate(".") 

using SynapseElife, Random, PiecewiseDeterministicMarkovProcesses, Sundials, Statistics, Distributions

# 1. OPTIMISATION : Charger le DataFrame UNE SEULE FOIS au démarrage
const DATA_PROTOCOL = dataProtocol("TigaretMellor16")
const INDICES_PROTOCOLES = [1, 2, 3, 4, 5, 6, 7]
const N_SWEEPS = 10

function evolveSynapse_light(xc0::Vector{𝒯}, xd0, p_synapse::SynapseParams,
                              events_sorted_times,
                              is_pre_or_post_event,
                              bap_by_epsp,
                              is_glu_released,
                              algos;
                              verbose = false, progress = false,
                              abstol = 1e-8, reltol = 1e-7, 
                              save_positions::Tuple{Bool, Bool} = (false, true),
                              nu = buildTransitionMatrix(), kwargs...) where 𝒯

	save_positionsON = save_positionsOFF = save_positions

	@assert eltype(is_pre_or_post_event) == Bool "Provide booleans for glutamate releases."
	@assert eltype(is_glu_released) == Bool "Provide booleans for glutamate indices."

	# we collect which external events correspond to BaPs
	events_bap = events_sorted_times[is_pre_or_post_event .== false]

	# function to simulate the synapse when Glutamate is ON
	SimGluON = (xc, xd, t1, t2, glu) -> SynapseElife.pdmpsynapse(xc, xd, t1, t2, events_bap, bap_by_epsp, glu, p_synapse, nu; algo = algos[1], save_positions = save_positionsON, reltol, abstol, kwargs...)

	# function to simulate the synapse when Glutamate is OFF
	SimGluOFF = (xc, xd, t1, t2)	 -> SynapseElife.pdmpsynapse(xc, xd, t1, t2, events_bap, bap_by_epsp, zero(𝒯), p_synapse, nu;  algo = algos[2], save_positions = save_positionsOFF, reltol, abstol, kwargs...)

	# random variable for Glutamate concentration
	gluDist = Gamma(1/p_synapse.glu_cv^2, p_synapse.glu_cv^2)

	current_xc = copy(xc0)
	current_xd = copy(xd0)
	current_t = zero(𝒯)

	# we loop over the external events
	for (eveindex, eve) in enumerate(events_sorted_times)
		if is_pre_or_post_event[eveindex] == true # it is a pre-synaptic event
			# simulate the event with Glutamate OFF
			res = SimGluOFF(current_xc, current_xd, current_t, eve)
			current_xc = res.xc[:, end]
			current_xd = res.xd[:, end]
			current_t = eve
			
			gluamp = rand(gluDist)
			# simulate the event with Glutamate ON
			res = SimGluON(current_xc, current_xd, eve, eve + p_synapse.glu_width, ifelse(is_glu_released[eveindex], gluamp, zero(𝒯)))
			current_xc = res.xc[:, end]
			current_xd = res.xd[:, end]
			current_t = eve + p_synapse.glu_width
		end
	end

	# reaching tend
	res = SimGluOFF(current_xc, current_xd, current_t, p_synapse.t_end)
	current_xd = res.xd[:, end]

	return (XD = reshape(current_xd, :, 1),)
end

function simuler_synapse_brute(val_n_ampa, val_n_nmda, val_n_caT, val_n_caR, val_n_caL, val_l_neck, k)
    # Limites de sécurité strictes (pour éviter les synapses "vides" ou aberrantes)
    val_l_neck = max(val_l_neck, 0.05) 
    val_n_ampa = max(1.0, val_n_ampa)
    val_n_nmda = max(1.0, val_n_nmda)
    val_n_caT  = max(1.0, val_n_caT)
    val_n_caR  = max(1.0, val_n_caR)
    val_n_caL  = max(1.0, val_n_caL)

    # ---------------------------------------------------------
    # BLOC TRY-CATCH : Intercepte les crashs du solveur ODE/PDMP
    # ---------------------------------------------------------
    try
        start = 0.5e3

        events_times, is_pre_or_post_event = firingPattern(
            start_time   = start,
            n_pos        = DATA_PROTOCOL[!,:n_pos][k],
            delay_pos    = DATA_PROTOCOL[!,:delay_pos][k],
            n_pre        = DATA_PROTOCOL[!,:n_pre][k],
            delay_pre    = DATA_PROTOCOL[!,:delay_pre][k],
            delay        = DATA_PROTOCOL[!,:delay][k],
            pulse        = DATA_PROTOCOL[!,:pulse][k],
            freq         = DATA_PROTOCOL[!,:freq][k],
            causal       = DATA_PROTOCOL[!,:causal][k],
            repeat_times = DATA_PROTOCOL[!,:repeat_times][k],
            repeat_after = DATA_PROTOCOL[!,:repeat_after][k]
        )

        param_synapse = SynapseParams(
            t_end         = isempty(events_times) ? start + 500.0 : events_times[end] + 500.0,
            soma_dist     = 200.0,
            temp_rates    = DATA_PROTOCOL[!,:temp][k],
            Ca_ext        = DATA_PROTOCOL[!,:exca][k],
            Mg            = DATA_PROTOCOL[!,:exmg][k],
            age           = DATA_PROTOCOL[!,:age][k],
            injbap        = DATA_PROTOCOL[!,:inj_time][k],
            I_clamp       = DATA_PROTOCOL[!,:injection][k],
            sampling_rate = 10.0,
            N_ampa = round(Int, val_n_ampa),
            N_NMDA = round(Int, val_n_nmda),
            N_caT  = round(Int, val_n_caT),
            N_caR  = round(Int, val_n_caR),
            N_caL  = round(Int, val_n_caL),
            L_neck = val_l_neck
        )

        xc0 = initial_conditions_continuous_temp(param_synapse)
        xd0 = initial_conditions_discrete(param_synapse)

        is_glu_release, _, _, _, _, bap_by_epsp_times = stp(
            param_synapse.t_end, PreSynapseParams(h=0),
            events_times, is_pre_or_post_event, algo=CHV(CVODE_BDF())
        )

        valid_indices = events_times .< param_synapse.t_end
        events_valides = events_times[valid_indices]
        pre_post_valides = is_pre_or_post_event[valid_indices]
        glu_valides = is_glu_release[valid_indices]

        result = evolveSynapse_light(
            xc0, xd0, param_synapse,
            events_valides,
            pre_post_valides, Float64[], glu_valides,
            (CHV(CVODE_BDF(linear_solver=:GMRES)), CHV(CVODE_BDF(linear_solver=:GMRES)));
            abstol = 1e-6, reltol = 1e-5, save_positions = (false, true), save_everystep = false, verbose = false
        )

        NC_final  = result.XD[36, end]
        LTD_final = result.XD[37, end]
        LTP_final = result.XD[38, end]

        w_NC  = 1.0
        w_LTD = 0.3
        w_LTP = 3.5

        delta_W = (NC_final * w_NC + LTD_final * w_LTD + LTP_final * w_LTP) / 100.0

        return delta_W

    catch e
        println("\n=== ERREUR JULIA DÉTECTÉE ===")
        Base.showerror(stdout, e) # Ceci va imprimer "UndefVarError: [NOM_DE_LA_VARIABLE] not defined"
        println("\n=============================\n")
        return NaN
    end
end

# -------------------------------------------------------------
# PATRON DE CONCEPTION : STRATÉGIE (Strategy Pattern)
# -------------------------------------------------------------
# Nous utilisons le patron de conception "Stratégie" combiné au dispatch
# multiple de Julia pour définir et interchanger les modes d'exécution.
# Cela permet d'avoir un unique point d'entrée `simulateur_sbi` tout en gardant
# une séparation nette entre l'exécution monothread (séquentielle) et multithread (parallèle).
abstract type ExecutionStrategy end
struct SequentialStrategy <: ExecutionStrategy end
struct ParallelStrategy <: ExecutionStrategy end

# 1. Stratégie Parallèle (Multi-thread plat sur l'ensemble du lot)
function simulateur_sbi(::ParallelStrategy, batch_params; nb_protocoles=7, n_sweeps=N_SWEEPS, start_sim_idx=1)
    # Conversion en types natifs Julia pour éviter le lock GIL de Python dans les threads
    native_batch = [Vector{Float64}(p) for p in batch_params]
    num_thetas = length(native_batch)
    println("\n=== [Parallel Sim] Début du lot avec $num_thetas jeux de paramètres (Simulations #$start_sim_idx à #$(start_sim_idx + num_thetas - 1)) ===")
    
    # Chaque tâche est définie par (theta_idx, sweep_idx, protocole_idx)
    tasks = [(t_idx, i, k) for t_idx in 1:num_thetas for i in 1:n_sweeps for k in 1:nb_protocoles]
    total_tasks = length(tasks)
    
    changements_de_poids = zeros(Float64, n_sweeps, nb_protocoles, num_thetas)
    tasks_done = Threads.Atomic{Int}(0)
    
    # Parallélisation plate sur l'ensemble de toutes les simulations de tous les thetas
    Threads.@threads for (t_idx, i, k) in tasks
        params = native_batch[t_idx]
        if length(params) == 4
            val_ampa, val_nmda, val_ca, val_neck = params
            val_caT = val_ca
            val_caR = val_ca
            val_caL = val_ca
        else
            val_ampa, val_nmda, val_caT, val_caR, val_caL, val_neck = params
        end
        
        changements_de_poids[i, k, t_idx] = simuler_synapse_brute(val_ampa, val_nmda, val_caT, val_caR, val_caL, val_neck, k)
        
        c = Threads.atomic_add!(tasks_done, 1) + 1
        global_sim_idx = start_sim_idx + t_idx - 1
        println("      [Parallel Sim] Sim #$global_sim_idx : Tâche $c/$total_tasks (Protocole $k, Sweep $i) traitée...")
    end
    
    # Regroupement et affichage des simulations terminées
    resultats = regrouper_resultats(changements_de_poids, num_thetas, nb_protocoles)
    for t_idx in 1:num_thetas
        global_sim_idx = start_sim_idx + t_idx - 1
        has_nan = any(isnan.(resultats[t_idx]))
        if has_nan
            println("   ⚠️ [Parallel Sim] Sim #$global_sim_idx terminée avec des échecs (contient NaN).")
        else
            println("   ✓ [Parallel Sim] Sim #$global_sim_idx terminée avec succès (tous les protocoles valides).")
        end
    end
    
    return resultats
end

# 2. Stratégie Séquentielle (Mono-thread)
function simulateur_sbi(::SequentialStrategy, batch_params; nb_protocoles=7, n_sweeps=N_SWEEPS, start_sim_idx=1)
    native_batch = [Vector{Float64}(p) for p in batch_params]
    num_thetas = length(native_batch)
    println("\n=== [Sequential Sim] Début du lot avec $num_thetas jeux de paramètres (Simulations #$start_sim_idx à #$(start_sim_idx + num_thetas - 1)) ===")
    
    changements_de_poids = zeros(Float64, n_sweeps, nb_protocoles, num_thetas)
    total_tasks = num_thetas * n_sweeps * nb_protocoles
    c = 0
    
    for t_idx in 1:num_thetas
        global_sim_idx = start_sim_idx + t_idx - 1
        params = native_batch[t_idx]
        if length(params) == 4
            val_ampa, val_nmda, val_ca, val_neck = params
            val_caT = val_ca
            val_caR = val_ca
            val_caL = val_ca
        else
            val_ampa, val_nmda, val_caT, val_caR, val_caL, val_neck = params
        end
        
        for k in 1:nb_protocoles
            for i in 1:n_sweeps
                changements_de_poids[i, k, t_idx] = simuler_synapse_brute(val_ampa, val_nmda, val_caT, val_caR, val_caL, val_neck, k)
                c += 1
                println("      [Sequential Sim] Sim #$global_sim_idx : Tâche $c/$total_tasks (Protocole $k, Sweep $i) traitée...")
            end
        end
    end
    
    resultats = regrouper_resultats(changements_de_poids, num_thetas, nb_protocoles)
    for t_idx in 1:num_thetas
        global_sim_idx = start_sim_idx + t_idx - 1
        has_nan = any(isnan.(resultats[t_idx]))
        if has_nan
            println("   ⚠️ [Sequential Sim] Sim #$global_sim_idx terminée avec des échecs (contient NaN).")
        else
            println("   ✓ [Sequential Sim] Sim #$global_sim_idx terminée avec succès.")
        end
    end
    
    return resultats
end

# Fonction utilitaire partagée pour le calcul des statistiques (Façade interne)
function regrouper_resultats(changements_de_poids, num_thetas, nb_protocoles)
    resultats = [Float64[] for _ in 1:num_thetas]
    for t_idx in 1:num_thetas
        for k in 1:nb_protocoles
            protocol_sweeps = changements_de_poids[:, k, t_idx]
            sweeps_valides = filter(!isnan, protocol_sweeps)
            
            if length(sweeps_valides) < 2
                push!(resultats[t_idx], NaN, NaN)
            else
                push!(resultats[t_idx], mean(sweeps_valides), var(sweeps_valides))
            end
        end
    end
    GC.gc()
    return resultats
end
