function nn_law(ε)
    ae(ε, [20,20,20,3])
end

function nn_nonassociated_plasticity(epsilon, epsilon_n, sigma)
    coef = ae([epsilon epsilon_n sigma], [20,20,20,6])
    coef = [coef constant(ones(size(coef,1),1))]
    H = consistent_tangent_matrix(coef, Dc)
    return batch_matmul(H, epsilon-epsilon_n) + sigma
end

function nn_associated_plasticity(epsilon, epsilon_n, sigma)
    
    coef = ae([epsilon epsilon_n sigma], [20,20,20,3])

    coef = [coef coef constant(ones(size(coef,1),1))]
    H = consistent_tangent_matrix(coef, Dc)
    return batch_matmul(H, epsilon-epsilon_n) + sigma
end