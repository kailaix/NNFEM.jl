function DebugDynamicMatLawLoss(domain::Domain, E_all::Array{Float64}, w∂E∂u_all::Array{Float64},
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
       
       fint, σ = AssembleInternalForce(domain,nn,E,DE,w∂E∂u,σ0)
       
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
   stack(out)[2:NT]
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
       

       fint, σ = AssembleInternalForce(domain,nn,E,DE,w∂E∂u,σ0)

       current_loss = sum((fint-F_tot[i-1])^2)
       op = tf.print(i, current_loss, summarize=-1)
       ta_loss = write(ta_loss, i, current_loss)
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

function DebugDynamicMatLawLoss(domain::Domain, globdat::GlobalData, state_history::Array{T}, fext_history::Array{S}, nn::Function, Δt::Float64) where {T, S}
    # todo convert to E_all, Ftot
    domain.history["state"] = state_history
    F_tot, E_all, w∂E∂u_all = preprocessing(domain, globdat, hcat(fext_history...), Δt)
    DebugDynamicMatLawLoss(domain, E_all, w∂E∂u_all, F_tot, nn)
end