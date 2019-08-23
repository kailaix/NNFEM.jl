export DynamicMatLawLoss, preprocessing, fitlinear

@doc """
    domain   : finite element domain, for data structure
    E_all    : all strains for the whole simulation, with size (NT+1, neles*nGauss, nstrains)
    w∂E∂u_all: multiplication of the Gaussian weight and ∂E∂u^T for the whole simulation, 
               with size (NT+1, neles*nGauss, ndofs_per_element, nstrains)
    F_tot : approximated internal force for the whole simulation, with size(NT, ndofs), 
            from time n=1 to time n=NT

    form the loss function, based on dynamic equilibrium 
        (Mddu + fint(NN, E, DE) + MIDddu_bc = fext

    loss = ∑ ||fint(NN, E, DE) - (fext - MIDddu_bc - Mddu)||^2
"""->
function DynamicMatLawLoss(domain::Domain, E_all::Array{Float64}, w∂E∂u_all::Array{Float64},
     F_tot::Array{Float64}, nn::Function)
    # todo, use fint instead of computed F_tot 
    # F_tot =  hcat(domain.history["fint"]...)'
    # define variables
    neles = domain.neles
    nGauss = length(domain.elements[1].weights)
    nstrains = size(E_all,3)

    NT = size(E_all,1)-1
    @assert size(E_all)==(NT+1, neles*nGauss, nstrains)
    @assert size(F_tot)==(NT, domain.neqs)
    # @show E_all[2,:,:]
    E_all = constant(E_all)
    F_tot = constant(F_tot)
    w∂E∂u_all = constant(w∂E∂u_all)

    function cond0(i, ta_loss, ta_σ)
        i<=NT+1
        # i<=2
    end

    function body(i, ta_loss, ta_σ)
        E = E_all[i]
        DE = E_all[i-1]
        w∂E∂u = w∂E∂u_all[i]
        σ0 = read(ta_σ, i-1)
        
        fint, σ = tfAssembleInternalForce(domain,nn,E,DE,w∂E∂u,σ0)
        
        # op = tf.print(i, fint, summarize=-1)
        # fint = bind(fint, op)

        # op = tf.print(i, F_tot[i-1], summarize=-1)
        # fint = bind(fint, op)

        # op = tf.print("F_tot",F_tot[i-1], summarize=-1)
        # i = bind(i, op)


        ta_σ = write(ta_σ, i, σ)
        ta_loss = write(ta_loss, i, sum((fint-F_tot[i-1])^2))
        i+1, ta_loss, ta_σ
    end

    σ0 = constant(zeros(neles*nGauss, nstrains))
    ta_σ = TensorArray(NT+1); ta_σ = write(ta_σ, 1, σ0)
    ta_loss = TensorArray(NT+1); ta_loss = write(ta_loss, 1, constant(0.0))
    i = constant(2, dtype=Int32)
    _, out, _ = while_loop(cond0, body, [i,ta_loss, ta_σ]; parallel_iterations=20)
    total_loss = sum(stack(out)[2:NT])
    return total_loss
end


@doc """
    for debugging
"""->
function DynamicMatLawLoss(domain::Domain, E_all::Array{Float64}, w∂E∂u_all::Array{Float64},
    F_tot::Array{Float64}, σ_all::Array{Float64}, nn::Function)
   F_tot =  hcat(domain.history["fint"]...)'
   # define variables
   neles = domain.neles
   nGauss = length(domain.elements[1].weights)
   nstrains = size(E_all,3)

   NT = size(E_all,1)-1
   @assert size(E_all)==(NT+1, neles*nGauss, nstrains)
   @assert size(F_tot)==(NT, domain.neqs)
#    println( size(σ_all))
   @assert size(σ_all)==(NT+1, neles*nGauss, nstrains)
   # @show E_all[2,:,:]
   E_all = constant(E_all)
   F_tot = constant(F_tot)
   σ_all = constant(σ_all)
   w∂E∂u_all = constant(w∂E∂u_all)

   function cond0(i, ta_loss, ta_σ)
       i<=NT+1
   end

   function body(i, ta_loss, ta_σ)
       E = E_all[i]
       DE = E_all[i-1]
       w∂E∂u = w∂E∂u_all[i]
       
    #    σ0 = σ_all[i-1] # ! for debugging 
       σ0 = read(ta_σ, i-1)
       

       fint, σ = tfAssembleInternalForce(domain,nn,E,DE,w∂E∂u,σ0)
    #    σ_ = nn(E, DE, σ0)

        #    σ ≈ σ_all[i]

       # op = tf.print(i, fint, summarize=-1)
       # fint = bind(fint, op)
       # op = tf.print([E, DE, σ0, σ], summarize=-1)
       # σ = bind(σ, op)

       current_loss = sum((fint-F_tot[i-1])^2)
       op = tf.print(i, current_loss, summarize=-1)
    #    i = bind(i, op)
       ta_loss = write(ta_loss, i, current_loss)
    #    op1 = tf.print(i-1, "fint",   fint, summarize=-1)
    #    op2 = tf.print(i-1, "F_tot",  F_tot[i-1], summarize=-1)
       op3 = tf.print(i-1, "correct σ",   σ_all[i]', summarize=-1)
       op4 = tf.print(i-1, "σ",   σ', summarize=-1)
    #    op5 = tf.print(i-1, "E",   E, summarize=-1)
    #    op6 = tf.print(i-1, "DE",   DE, summarize=-1)
    #    op7 = tf.print(i-1, "w∂E∂u",   w∂E∂u, summarize=-1)
    #    op8 = tf.print(i-1, "σ_",   σ_', summarize=-1)
       
       
    #    i = bind(i, op1)
    #    i = bind(i, op2)
    #    i = bind(i, op3)
    #    i = bind(i, op4)
    #    i = bind(i, op5)
    #    i = bind(i, op6)
    #    i = bind(i, op7)
    #    i = bind(i, op8)
       
       ta_σ = write(ta_σ, i, σ)
       i+1, ta_loss, ta_σ
   end

   σ0 = constant(zeros(neles*nGauss, nstrains))
   ta_loss = TensorArray(NT+1); ta_loss = write(ta_loss, 1, constant(0.0))
   ta_σ = TensorArray(NT+1); ta_σ = write(ta_σ, 1, σ0)
   i = constant(2, dtype=Int32)
   _, out = while_loop(cond0, body, [i,ta_loss,ta_σ]; parallel_iterations=20)
   total_loss = sum(stack(out)[2:NT])
   return total_loss
end

@doc """
    domain   : finite element domain
    globdat  : finite element data structure
    state_history : displace history of all time steps and all nodes, 
                    a list of NT+1  ndof-displacement vectors, including time 0
                    hcat(state_history...) gives a matrix of size(ndof-displacement, NT+1)
    fext_history  : external force load of all time steps and all nodes, 
                    a list of NT+1 ndof-external-force vectors, including time 0
                    hcat(fext_history...) gives a matrix of size(ndof-external-force, NT+1)
    nn: Neural network
    Δt: time step size
    
    compute loss function from state and external force history 
"""->
function DynamicMatLawLoss(domain::Domain, globdat::GlobalData, state_history::Array{T}, fext_history::Array{S}, nn::Function, Δt::Float64) where {T, S}
    # todo convert to E_all, Ftot
    domain.history["state"] = state_history
    F_tot, E_all, w∂E∂u_all = preprocessing(domain, globdat, hcat(fext_history...), Δt)
    DynamicMatLawLoss(domain, E_all, w∂E∂u_all, F_tot, nn)
end

function DynamicMatLawLoss(domain::Domain, globdat::GlobalData, state_history::Array{T}, fext_history::Array{S}, nn::Function, Δt::Float64, n::Int64) where {T, S}
    domain.history["state"] = state_history
    F_tot, E_all, w∂E∂u_all = preprocessing(domain, globdat, hcat(fext_history...), Δt, n)
    DynamicMatLawLoss(domain, E_all, w∂E∂u_all, F_tot, nn)
end

@doc """
    compute F_tot ≈ F_int , ane E_all
"""->
function preprocessing(domain::Domain, globdat::GlobalData, F_ext::Array{Float64},Δt::Float64)
    U = hcat(domain.history["state"]...)
    # @show size(U)
    # @info " U ", size(U),  U'
    M = globdat.M
    MID = globdat.MID 

    NT = size(U,2)-1

    #Acceleration of Dirichlet nodes
    bc_acc = zeros(sum(domain.EBC.==-2),NT)
    if !(globdat.EBC_func===nothing)
        for i = 1:NT
            _, bc_acc[:,i]  = globdat.EBC_func(Δt*i)
        end
    end
    
    ∂∂U = zeros(size(U,1), NT+1)
    ∂∂U[:,2:NT] = (U[:,1:NT-1]+U[:,3:NT+1]-2U[:,2:NT])/Δt^2
    # @show size(∂∂U),size(U)
    if size(F_ext,2)==NT+1
        F_tot = F_ext[:,2:end] - M*∂∂U[domain.dof_to_eq,2:end] - MID*bc_acc
    elseif size(F_ext,2)==NT
        # @show size(∂∂U)
        F_tot = F_ext - M*∂∂U[domain.dof_to_eq,2:end] - MID*bc_acc
    else
        error("F size is not valid")
    end
    
    neles = domain.neles
    nGauss = length(domain.elements[1].weights)
    neqns_per_elem = length(getEqns(domain,1))

    
    nstrains = div((domain.elements[1].eledim + 1)*domain.elements[1].eledim, 2)

    E_all = zeros(NT+1, neles*nGauss, nstrains)
    w∂E∂u_all = zeros(NT+1, neles*nGauss, neqns_per_elem, nstrains)

    for i = 1:NT+1
        domain.state = U[:, i]
        # @info "domain state", domain.state
        # Loop over the elements in the elementGroup to construct strain and geo-matrix E_all and w∂E∂u_all
        for iele  = 1:neles
            element = domain.elements[iele]

            # Get the element nodes
            el_nodes = getNodes(element)

            # Get the element nodes
            el_eqns = getEqns(domain,iele)

            el_dofs = getDofs(domain,iele)

            el_state  = getState(domain,el_dofs)

            # Get the element contribution by calling the specified action
            E, w∂E∂u = getStrain(element, el_state) 
            # if i==2
            #     @info (el_state, E)
            # end
      
            # @show E, nGauss
            E_all[i, (iele-1)*nGauss+1:iele*nGauss, :] = E

            w∂E∂u_all[i, (iele-1)*nGauss+1:iele*nGauss,:,:] = w∂E∂u
        end
    end
    @info "preprocessing end..."
    # # DEBUG
    # fext = hcat(domain.history["fint"]...)
    return F_tot'|>Array, E_all, w∂E∂u_all
end


@doc """
preprocessing(domain::Domain, globdat::GlobalData, F_ext::Array{Float64},Δt::Float64, n::Int64)

Same as `preprocessing`, except that only the first `n` steps are considered
"""->
function preprocessing(domain::Domain, globdat::GlobalData, F_ext::Array{Float64},Δt::Float64, n::Int64)
    U = hcat(domain.history["state"]...)
    # @info " U ", size(U),  U'
    M = globdat.M
    MID = globdat.MID 

    NT = size(U,2)-1

    #Acceleration of Dirichlet nodes
    bc_acc = zeros(sum(domain.EBC.==-2),NT)
    for i = 1:NT
        _, bc_acc[:,i]  = globdat.EBC_func(Δt*i)
    end

    
    ∂∂U = zeros(size(U,1), NT+1)
    ∂∂U[:,2:NT] = (U[:,1:NT-1]+U[:,3:NT+1]-2U[:,2:NT])/Δt^2
 
    if size(F_ext,2)==NT+1
        F_tot = F_ext[:,2:end] - M*∂∂U[domain.dof_to_eq,2:end] - MID*bc_acc
    elseif size(F_ext,2)==NT
        F_tot = F_ext - M*∂∂U[domain.dof_to_eq,2:end] - MID*bc_acc
    else
        error("F size is not valid")
    end
    
    neles = domain.neles
    nGauss = length(domain.elements[1].weights)
    neqns_per_elem = length(getEqns(domain,1))

    
    nstrains = div((domain.elements[1].eledim + 1)*domain.elements[1].eledim, 2)

    E_all = zeros(NT+1, neles*nGauss, nstrains)
    w∂E∂u_all = zeros(NT+1, neles*nGauss, neqns_per_elem, nstrains)

    for i = 1:NT+1
        domain.state = U[:, i]
        # @info "domain state", domain.state
        # Loop over the elements in the elementGroup to construct strain and geo-matrix E_all and w∂E∂u_all
        for iele  = 1:neles
            element = domain.elements[iele]

            # Get the element nodes
            el_nodes = getNodes(element)

            # Get the element nodes
            el_eqns = getEqns(domain,iele)

            el_dofs = getDofs(domain,iele)

            el_state  = getState(domain,el_dofs)

            # Get the element contribution by calling the specified action
            E, w∂E∂u = getStrain(element, el_state) 
            # if i==2
            #     @info (el_state, E)
            # end
      
            # @show E, nGauss
            E_all[i, (iele-1)*nGauss+1:iele*nGauss, :] = E

            w∂E∂u_all[i, (iele-1)*nGauss+1:iele*nGauss,:,:] = w∂E∂u
        end
    end
    @info "preprocessing end..."
    # # DEBUG
    # fext = hcat(domain.history["fint"]...)
    F_tot = F_tot'|>Array

    F_tot[1:n,:], E_all[1:n+1, :, :], w∂E∂u_all[1:n+1, :, :, :]
end

function fitlinear(n::Int64, domain::Domain, globdat::GlobalData,
         n_data::Int64, Δt::Float64, wkdir::String, stress_scale::Float64)
    A = Variable(rand(3,3))
    H = A + A'
    # H = constant([2.50784e11  1.12853e11  0.0       
    # 1.12853e11  2.50784e11  0.0       
    # 0.0         0.0         6.89655e10]/stress_scale)
    function linear_map(ε, ε0, σ0)
        y = ε*H*stress_scale
    end

    losses = Array{PyObject}(undef, n_data)
    for i = 1:n_data
        state_history, fext_history = read_data("$wkdir/Data/$i.dat")
        losses[i] = DynamicMatLawLoss(domain, globdat, state_history, fext_history, linear_map,Δt, n)
    end
    loss = sum(losses)/stress_scale
    sess = Session(); init(sess)
    @show run(sess, loss)
    BFGS!(sess, loss, 2000)
    run(sess, H)*stress_scale
end