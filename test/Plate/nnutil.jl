
using ForwardDiff
using DelimitedFiles

function get_matrix(o::PyObject)
    tensor([o[1] o[2] o[3];
    o[2] o[4] o[5];
    o[3] o[5] o[6]])
end

function get_matrix(o::AbstractArray)
    [o[1] o[2] o[3];
    o[2] o[4] o[5];
    o[3] o[5] o[6]]
end

function nn(ε, ε0, σ0) # ε, ε0, σ0 are all length 3 vector
    local y, H

    if nntype=="linear"
        y = ε*H0*stress_scale
        y
    elseif nntype=="linear_test"
        y = ε*HH
        y 
    elseif nntype=="ae_scaled"
        x = [ε/strain_scale ε0/strain_scale σ0/stress_scale]
        if isa(x, Array)
            x = constant(x)
        end
        y = ae(x, [20,20,20,20,3], "ae_scaled")*stress_scale
    elseif nntype=="mae"
        x = [ε/strain_scale ε0/strain_scale σ0/stress_scale]
        if isa(x, Array)
            x = constant(x)
        end
        y = ae(x, [20,20,20,20,9], "mae")*stress_scale
        out = tf.map_fn(x->squeeze(reshape(x[2],1,3)*get_matrix(x[1])), (y, ε/strain_scale), dtype=tf.float64)
        out
    end

end



function sigmoid_(z)

    return 1.0 / (1.0 + exp(-z))
  
end



function nn_helper(ε, ε0, σ0)
    if nntype=="ae_scaled"
        x = reshape([ε;ε0;σ0/stress_scale],1, 9)
        reshape(nnae_scaled(x)*stress_scale,3,1)
    elseif nntype=="linear"
        x = reshape(reshape(ε,1,3)*H0,3,1)
    elseif nntype=="mae"
        x = reshape([ε/strain_scale;ε0/strain_scale;σ0/stress_scale],1, 9)
        y = reshape(ε/strain_scale, 1, 3)*get_matrix(nnae_scaled(x))*stress_scale
        reshape(y, 3, 1)
    end
end

function post_nn(ε, ε0, σ0, Δt)
    # @show "Post NN"
    f = x -> nn_helper(x, ε0, σ0)
    df = ForwardDiff.jacobian(f, ε)
    return f(ε), df
end

function check_grad()
    ε = rand(3); ε0 = rand(3); σ0 = rand(3)*1e10; Δt = 0.0
    y_, J = post_nn(ε, ε0, σ0, Δt)

    v_ = rand(3)
    ms_ = Array{Array{Float64}}(undef, 5)
    ys_ = Array{Array{Float64}}(undef, 5)
    sval_ = Array{Float64}(undef, 5)
    wval_ = Array{Float64}(undef, 5)
    gs_ =  @. 1 / 10^(1:5)

    for i = 1:5
        g_ = gs_[i]
        ms_[i] = ε + g_*v_
        ys_[i],_ = post_nn(ms_[i], ε0, σ0, Δt)
        sval_[i] = norm(ys_[i] - y_)
        wval_[i] = norm(ys_[i] - y_ - g_*J*v_)
    end


    close("all")
    loglog(gs_, abs.(sval_), "*-", label="finite difference")
    loglog(gs_, abs.(wval_), "+-", label="automatic differentiation")
    loglog(gs_, gs_.^2 * 0.5*abs(wval_[1])/gs_[1]^2, "--",label="\$\\mathcal{O}(\\gamma^2)\$")
    loglog(gs_, gs_ * 0.5*abs(sval_[1])/gs_[1], "--",label="\$\\mathcal{O}(\\gamma)\$")

    plt.gca().invert_xaxis()
    legend()
    xlabel("\$\\gamma\$")
    ylabel("Error")

end