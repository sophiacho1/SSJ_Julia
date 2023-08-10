using LinearAlgebra, Interpolations, PyPlot


function discretize_assets(amin, amax, n_a)
    # find maximum ubar of uniform grid corresponding to desired maximum amax of asset grid
    ubar = log(1 + log(1 + amax - amin))
    
    # make uniform grid
    u_grid = LinRange(0, ubar, n_a)
    
    # double-exponentiate uniform grid and add amin to get grid from amin to amax
    return amin .+ exp.(exp.(u_grid) .- 1) .- 1
end

function rouwenhorst_Pi(N, p)
    # base case Pi_2
    Pi = [p 1 - p; 1 - p p]
    
    # recursion to build up from Pi_2 to Pi_N
    for n in 3:N
        Pi_old = Pi
        Pi = zeros(n, n)
        
        Pi[1:end-1, 1:end-1] .+= p .* Pi_old
        Pi[1:end-1, 2:end] .+= (1 - p) .* Pi_old
        Pi[2:end, 1:end-1] .+= (1 - p) .* Pi_old
        Pi[2:end, 2:end] .+= p .* Pi_old
        Pi[2:end-1, :] ./= 2
    end
    
    return Pi
end


function stationary_markov(Pi, tol=1E-14)
    # start with uniform distribution over all states
    n = size(Pi, 1)
    pi = fill(1/n, n)
    
    # update distribution using Pi until successive iterations differ by less than tol
    for _ in 1:10000
        pi_new = Pi' * pi
        if maximum(abs.(pi_new - pi)) < tol
            return pi_new
        end
        pi = pi_new
    end
end

function discretize_income(rho, sigma, n_s)
    # choose inner-switching probability p to match persistence rho
    p = (1 + rho) / 2
    
    # start with states from 0 to n_s-1, scale by alpha to match standard deviation sigma
    s = collect(0:n_s-1)
    alpha = 2 * sigma / sqrt(n_s - 1)
    s = alpha .* s
    
    # obtain Markov transition matrix Pi and its stationary distribution
    Pi = rouwenhorst_Pi(n_s, p)
    pi = stationary_markov(Pi)
    
    # s is log income, get income y and scale so that mean is 1
    y = exp.(s)
    y /= dot(pi, y)
    
    return y, pi, Pi
end


function backward_iteration(Va, Pi, a_grid, y, r, beta, eis)
    # Step 1: Discounting and expectations
    Wa = (beta * Pi) * Va

    # Step 2: Solving for asset policy using the first-order condition
    c_endog = Wa .^ (-eis)
    coh = y .+ (1+r) .* a_grid'

    a = similar(coh, eltype(coh))
    for s in 1:length(y)
        interp = LinearInterpolation(a_grid .+ c_endog[s, :], a_grid, extrapolation_bc=Flat())
        a[s, :] = interp(coh[s, :])
    end

    # Step 3: Enforcing the borrowing constraint and backing out consumption
    a = max.(a, a_grid[1])
    c = coh .- a

    # Step 4: Using the envelope condition to recover the derivative of the value function
    Va = (1+r) .* c .^ (-1/eis)

    return Va, a, c
end


function policy_ss(Pi, a_grid, y, r, beta, eis, tol=1E-9)
    # initial guess for Va: assume consumption 5% of cash-on-hand, then get Va from envelope condition
    coh = y .+ (1 + r) .* a_grid'
    c = 0.05 .* coh
    Va = (1 + r) .* c .^ (-1 / eis)
    
    # iterate until maximum distance between two iterations falls below tol, fail-safe max of 10,000 iterations
    a_old = similar(coh, eltype(coh))
    
    for it in 1:10_000
        Va, a, c = backward_iteration(Va, Pi, a_grid, y, r, beta, eis)
        
        # after iteration 0, can compare new policy function to old one
        if it > 1 && maximum(abs.(a .- a_old)) < tol
            return Va, a, c
        end
        
        copyto!(a_old, a)
    end
    
    error("Did not converge within the maximum number of iterations.")
end


# function get_lottery_helper(a, a_grid)
#     # step 1: find the i such that a' lies between gridpoints a_i and a_(i+1)
#     a_i = searchsortedfirst(a_grid, a) - 1 # NOTE: a_i is going to be +1 of what it would be in Python
    
#     # step 2: obtain lottery probabilities pi
#     if a_i == 0
#         a_pi = (a_grid[a_i+1] - a)/(a_grid[a_i+1] - a_grid[end])
#     else
#         a_pi = (a_grid[a_i+1] - a)/(a_grid[a_i+1] - a_grid[a_i])
#     end
    
#     return a_i, a_pi
# end


# function get_lottery(a, a_grid)
#     a_i = similar(a, eltype(a))
#     a_pi = similar(a, eltype(a))
#     for row in 1:size(a, 1) 
#         for column in 1:size(a, 2)
#             a_i_helper, a_pi_helper = get_lottery_helper(a[row, column], a_grid)
#             a_i[row, column] = a_i_helper
#             a_pi[row, column] = a_pi_helper
#         end
#     end
#     return Int.(a_i), a_pi
# end

function get_lottery_pt(a, a_grid)

    left_idx = -1

    for i in 1:length(a_grid) - 1 # can improve with binary search
        if a_grid[i] <= a && a_grid[i + 1] >= a
            left_idx = i
            break
        end
    end

    prob = (a - a_grid[left_idx + 1])/(a_grid[left_idx] - a_grid[left_idx + 1])

    return left_idx, prob

end


# not sure whether there is a nice broadcasting feature in julia
# but can repeat for all grid points

#=
Params:
    1. a here is the n_e by n_a matrix
=#

function get_lottery(a, a_grid)

    a_i = zeros(size(a)[1], size(a)[2])
    a_pi = zeros(size(a)[1], size(a)[2])

    for row in 1:size(a)[1]
        for col in 1:size(a)[2]
            a_opt = a[row, col]
            left, left_prob = get_lottery_pt(a_opt, a_grid)
            a_i[row, col] = left
            a_pi[row, col] = left_prob

        end
    end

    return a_i, a_pi
    
end


function forward_policy(D, a_i, a_pi)

    D_next = zeros(size(D)[1], size(D)[2])

    for e in 1:size(a_i)[1]
        for a in 1:size(a_i)[2]
            D_next[e, Int(a_i[e, a])] += a_pi[e, a] * D[e, a]
            D_next[e, Int(a_i[e, a] + 1)] += (1 - a_pi[e, a]) * D[e, a]
        end
    end

    return D_next

end


function forward_iteration(D, Pi, a_i, a_pi)
    Dend = forward_policy(D, a_i, a_pi)
    return transpose(Pi) * Dend
end


function policy_ss(Pi, a_grid, y, r, beta, eis, tol=1E-9)
    # initial guess for Va: assume consumption 5% of cash-on-hand, then get Va from envelope condition
    coh = y .+ (1 + r) .* a_grid'
    c = 0.05 .* coh
    Va = (1 + r) .* c .^ (-1 / eis)
    
    # iterate until maximum distance between two iterations falls below tol, fail-safe max of 10,000 iterations
    a_old = similar(coh, eltype(coh))
    
    for it in 1:10_000
        Va, a, c = backward_iteration(Va, Pi, a_grid, y, r, beta, eis)
        
        # after iteration 0, can compare new policy function to old one
        if it > 1 && maximum(abs.(a .- a_old)) < tol
            return Va, a, c
        end
        
        copyto!(a_old, a)
    end
    
    error("Did not converge within the maximum number of iterations.")
end


function distribution_ss(Pi, a, a_grid, tol=1e-10)
    a_i, a_pi = get_lottery(a, a_grid)
    
    # as initial D, use stationary distribution for s, plus uniform over a
    pi = stationary_markov(Pi)
    D = (pi / length(a_grid)) .* ones(size(a))
    
    # now iterate until convergence to acceptable threshold
    for _ in 1:10000
        D_new = forward_iteration(D, Pi, a_i, a_pi)
        if maximum(abs.(D_new .- D)) < tol
            return D_new
        end
        D = D_new
    end
    
    error("Did not converge within the maximum number of iterations.")
end


function steady_state(Pi, a_grid, y, r, beta, eis)
    Va, a, c = policy_ss(Pi, a_grid, y, r, beta, eis)
    a_i, a_pi = get_lottery(a, a_grid)
    D = distribution_ss(Pi, a, a_grid)
    
    return Dict("D" => D, 
                "Va" => Va, 
                "a" => a, 
                "c" => c, 
                "a_i" => a_i, 
                "a_pi" => a_pi,
                "A" => dot(a, D), 
                "C" => dot(c, D),
                "Pi" => Pi, 
                "a_grid" => a_grid, 
                "y" => y, 
                "r" => r, 
                "beta" => beta, 
                "eis" => eis)
end


function expectation_policy(Xend, a_i, a_pi)
    X = zeros(size(Xend))
    for s in 1:size(a_i, 1)
        for a in 1:size(a_i, 2)
            if a_i[s, a] == 0
                X[s, a] = a_pi[s, a] * Xend[s, end] + (1 - a_pi[s, a]) * Xend[s, Int(a_i[s, a]) + 1]
            else
                # expectation is pi(s,a)*Xend(s,i(s,a)) + (1-pi(s,a))*Xend(s,i(s,a)+1)
                X[s, a] = a_pi[s, a] * Xend[s, Int(a_i[s, a])] + (1 - a_pi[s, a]) * Xend[s, Int(a_i[s, a]) + 1]
            end
        end
    end

    return X
end


function expectation_iteration(X, Pi, a_i, a_pi)
    Xend = Pi * X
    return expectation_policy(Xend, a_i, a_pi)
end


# function expectation_vectors(X, a_i, a_pi, T)
#     # set up array of curlyEs and fill in first row with base case
#     curlyE = Array{Float64}(undef, T, size(X, 1), size(X, 2))
#     curlyE[1, :, :] = X
    
#     # recursively apply the law of iterated expectations
#     for j in 2:T
#         curlyE[j, :, :] = expectation_iteration(curlyE[j-1, :, :], a_i, a_pi)
#     end
    
#     return curlyE
# end


function expectation_functions(X, Pi, a_i, a_pi, T)
    # set up array of curlyEs and fill in first row with base case
    curlyE = Array{Float64}(undef, T, size(X, 1), size(X, 2))
    curlyE[1, :, :] = X
    
    # recursively apply the law of iterated expectations
    for j in 2:T
        curlyE[j, :, :] = expectation_iteration(curlyE[j-1, :, :], Pi, a_i, a_pi)
    end
    
    return curlyE
end


function example_caliberation()
    y, _, Pi = discretize_income(0.975, 0.7, 7)
    return Dict("a_grid" => discretize_assets(0, 10000, 500), 
                "y" => y,
                "Pi" => Pi,
                "r" => 0.01/4,
                "beta" => 1 - 0.08/4,
                "eis" => 1
            ) 
end
