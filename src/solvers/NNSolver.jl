export DynamicMatLawLoss, BFGS, preprocessing, ADAM
function DynamicMatLawLoss(domain::Domain, E_all::Array{Float64}, fext::Array{Float64},
        nn::Function)
    # define variables
    neles = domain.neles
    nGauss = length(domain.elements[1].weights)
    nstrains = 3
    NT = size(E_all,1)-1
    @assert size(E_all)==(NT+1, neles*nGauss, nstrains)
    @assert size(fext)==(NT, domain.neqs)
    E_all = constant(E_all)
    fext = constant(fext)

    function cond0(i, ta_loss, ta_σ)
        i<=NT+1
    end

    function body(i, ta_loss, ta_σ)
        σ0 = read(ta_σ, i-1)
        E = E_all[i]
        DE = E_all[i-1]
        # @show E, DE
        fint, σ = tfAssembleInternalForce(domain,nn,E,DE,σ0)
        # @show E, DE
        # op = tf.print(i)
        # σ = bind(σ, op)
        ta_σ = write(ta_σ, i, σ)
        ta_loss = write(ta_loss, i, sum((fint-fext[i-1])^2))
        i+1, ta_loss, ta_σ
    end

    σ0 = constant(zeros(neles*nGauss, nstrains))
    ta_σ = TensorArray(NT+1); ta_σ = write(ta_σ, 1, σ0)
    ta_loss = TensorArray(NT+1); ta_loss = write(ta_loss, 1, constant(0.0))
    i = constant(2, dtype=Int32)
    _, out, _ = while_loop(cond0, body, [i,ta_loss, ta_σ]; parallel_iterations=10)
    total_loss = sum(stack(out)[2:NT])
    return total_loss
end

function BFGS(sess::PyObject, loss::PyObject; kwargs...)
    __cnt = 0
    function print_loss(l)
        if mod(__cnt,1)==0
            println("iter $__cnt, current loss=",l)
        end
        __cnt += 1
    end
    __iter = 0
    function step_callback(rk)
        if mod(__iter,1)==0
            println("================ ITER $__iter ===============")
        end
        __iter += 1
    end
    opt = ScipyOptimizerInterface(loss, method="L-BFGS-B",options=Dict("maxiter"=> 30000, "ftol"=>1e-12, "gtol"=>1e-12))
    @info "Optimization starts..."
    ScipyOptimizerMinimize(sess, opt, loss_callback=print_loss, step_callback=step_callback, fetches=[loss])
end

function ADAM(sess::PyObject, loss::PyObject; kwargs...)
    opt = AdamOptimizer().minimize(loss)
    init(sess)
    for i = 1:10000
        l, _ = run(sess, [loss, opt])
        println("Iter $i, loss = $l")
    end
end

# compute E from U 
function preprocessing(domain::Domain, globdat::GlobalData, F::Array{Float64},Δt::Float64)
    U = hcat(domain.state_history...)
    M = globdat.M
    @info size(U)
    NT = size(U,2)-1
    @info NT
    @assert size(F,2)==NT+1[]
    ∂∂U = zeros(size(U,1), NT+1)
    ∂∂U[:,2:NT] = (U[:,1:NT-1]+U[:,3:NT+1]-2U[:,2:NT])/Δt^2
    fext = F[:,2:end]-M*∂∂U[domain.dof_to_eq,2:end]
    
    neles = domain.neles
    nGauss = length(domain.elements[1].weights)
    neqns_per_elem = length(getEqns(domain,1))
    nstrains = 3
    E_all = zeros(NT+1, neles*nGauss, nstrains)

    for i = 1:NT+1
        domain.state = U[:, i]
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
            # @show E, nGauss
            E_all[i, (iele-1)*nGauss+1:iele*nGauss, :] = E
        end
    end
    return fext'|>Array, E_all
end