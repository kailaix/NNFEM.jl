function nn_law(ε)
    ae(ε, [20,20,20,3])
end

function nn_nonassociated_plasticity(sigma, epsilon, epsilon_n)
    coef = ae([sigma epsilon epsilon_n], [20,20,20,7])
    H = consistent_tangent_matrix(coef, Dc)
    return batch_matmul(H, epsilon-epsilon_n) + sigma
end

function nn_associated_plasticity(sigma, epsilon, epsilon_n)
    coef = ae([sigma epsilon epsilon_n], [20,20,20,4])
    coef = [coef[:,1:3] coef]
    H = consistent_tangent_matrix(coef, Dc)
    return batch_matmul(H, epsilon-epsilon_n) + sigma
end