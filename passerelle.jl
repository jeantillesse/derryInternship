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
const N_SWEEPS = 10
const SIMULATION_COUNTER = Threads.Atomic{Int}(0)

function simulateur_complet_sbi_multithread(params; nb_protocoles=7, n_sweeps=N_SWEEPS, sim_idx=0)
    if length(params) == 4
        val_ampa, val_nmda, val_ca, val_neck = params
        val_caT = val_ca
        val_caR = val_ca
        val_caL = val_ca
    else
        val_ampa, val_nmda, val_caT, val_caR, val_caL, val_neck = params
    end
    
    vecteur_cible = Float64[]
    
    # Pre-allocation d'une matrice pour stocker les résultats : lignes = sweeps, colonnes = protocoles
    changements_de_poids = zeros(Float64, n_sweeps, nb_protocoles)
    
    # Création de la liste des tâches : chaque tâche est un couple (sweep_idx, protocole_idx)
    tasks = [(i, k) for i in 1:n_sweeps for k in 1:nb_protocoles]
    total_tasks = length(tasks)
    tasks_done = Threads.Atomic{Int}(0)
    
    # Parallélisation à plat sur l'ensemble des couples (sweep, protocole)
    Threads.@threads for (i, k) in tasks
        changements_de_poids[i, k] = simuler_synapse_brute(val_ampa, val_nmda, val_caT, val_caR, val_caL, val_neck, k)
        c = Threads.atomic_add!(tasks_done, 1) + 1
        println("      [Sim #$sim_idx] Tâche $c/$total_tasks : Protocole $k/$nb_protocoles, Sweep $i/$n_sweeps traité...")
    end
    
    # Traitement des résultats par protocole (sequentiel, très rapide car pas de simulation)
    for k in 1:nb_protocoles
        protocol_sweeps = changements_de_poids[:, k]
        sweeps_valides = filter(!isnan, protocol_sweeps)
        println("   ✓ [Sim #$sim_idx] Protocole $k/$nb_protocoles terminé : $(length(sweeps_valides))/$n_sweeps sweeps valides.")
        
        if length(sweeps_valides) < 2
            # S'il y a 0 ou 1 sweep valide, impossible de calculer une variance pertinente
            push!(vecteur_cible, NaN, NaN)
        else
            # On calcule les stats uniquement sur ce qui a fonctionné
            push!(vecteur_cible, mean(sweeps_valides), var(sweeps_valides))
        end
    end
    
    return vecteur_cible
end

function simulateur_complet_sbi_singlethread(params; nb_protocoles=7, n_sweeps=N_SWEEPS, sim_idx=0)
    if length(params) == 4
        val_ampa, val_nmda, val_ca, val_neck = params
        val_caT = val_ca
        val_caR = val_ca
        val_caL = val_ca
    else
        val_ampa, val_nmda, val_caT, val_caR, val_caL, val_neck = params
    end
    
    vecteur_cible = Float64[]
    
    for k in 1:nb_protocoles
        changements_de_poids = zeros(Float64, n_sweeps)
        
        # Boucle SÉQUENTIELLE sur les sweeps
        for i in 1:n_sweeps
            changements_de_poids[i] = simuler_synapse_brute(val_ampa, val_nmda, val_caT, val_caR, val_caL, val_neck, k)
            println("      [Sim #$sim_idx - Protocole $k/$nb_protocoles] Sweep $i/$n_sweeps traité...")
        end
        
        sweeps_valides = filter(!isnan, changements_de_poids)
        println("   ✓ [Sim #$sim_idx] Protocole $k/$nb_protocoles terminé : $(length(sweeps_valides))/$n_sweeps sweeps valides.")
        
        if length(sweeps_valides) < 2
            push!(vecteur_cible, NaN, NaN)
        else
            push!(vecteur_cible, mean(sweeps_valides), var(sweeps_valides))
        end
        # Libère la mémoire après chaque protocole
        GC.gc()
    end
    
    return vecteur_cible
end

function simulateur_complet_sbi(params; multithread=true, nb_protocoles=7, n_sweeps=N_SWEEPS)
    sim_idx = Threads.atomic_add!(SIMULATION_COUNTER, 1) + 1
    println("\n=== [Simulation #$sim_idx] Début avec paramètres : ", round.(params, digits=4), " ===")
    res = if multithread
        simulateur_complet_sbi_multithread(params; nb_protocoles=nb_protocoles, n_sweeps=n_sweeps, sim_idx=sim_idx)
    else
        simulateur_complet_sbi_singlethread(params; nb_protocoles=nb_protocoles, n_sweeps=n_sweeps, sim_idx=sim_idx)
    end
    # Libère la mémoire (trajectoires) après chaque simulation de batch
    GC.gc()
    return res
end

# Tu peux SUPPRIMER l'ancienne fonction "simulateur_batch_sbi", on n'en a plus besoin !

