using LinearAlgebra, Interpolations, PyPlot


"""Part 1: discretization tools"""

function discretize_assets(amin, amax, n_a)
    # Find maximum ubar of uniform grid corresponding to desired maximum amax of asset grid
    ubar = log(1 + log(1 + amax - amin))
    
    # Make uniform grid
    u_grid = LinRange(0, ubar, n_a)
    
    # Double-exponentiate uniform grid and add amin to get grid from amin to amax
    return amin .+ exp.(exp.(u_grid) .- 1) .- 1
end


function rouwenhorst_Pi(N, p)
    # Base case Pi_2
    Pi = [p 1 - p; 1 - p p]
    
    # Recursion to build up from Pi_2 to Pi_N
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
    # Start with uniform distribution over all states
    n = size(Pi, 1)
    pi = fill(1/n, n)
    
    # Update distribution using Pi until successive iterations differ by less than tol
    for _ in 1:10_000
        pi_new = Pi' * pi
        if maximum(abs.(pi_new - pi)) < tol
            return pi_new
        end
        pi = pi_new
    end
end


function discretize_income(rho, sigma, n_e)
    # Choose inner-switching probability p to match persistence rho
    p = (1 + rho) / 2
    
    # Start with states from 0 to n_e-1, scale by alpha to match standard deviation sigma
    e = collect(0:n_e - 1)
    alpha = 2 * sigma / sqrt(n_e - 1)
    e = alpha .* e
    
    # Obtain Markov transition matrix Pi and its stationary distribution
    Pi = rouwenhorst_Pi(n_e, p)
    pi = stationary_markov(Pi)
    
    # e is log income, get income y and scale so that mean is 1
    y = exp.(e)
    y /= dot(pi, y)
    
    return y, pi, Pi
end


"""Part 2: Backward iteration for policy"""
# function interpolate_y(x, xq, y)
#     yq = zeros(size(xq))
#     nxq, nx = size(xq, 1), size(x, 1)
#     xi = 1
#     x_low = x[1]
#     x_high = x[2]
#     for xqi_cur in 1:nxq
#         xq_cur = xq[xqi_cur]
#         while xi < nx - 1
#             if x_high >= xq_cur
#                 break
#             end
#             xi += 1
#             x_low = x_high
#             x_high = x[xi + 1]
#         end
#         xqpi_cur = (x_high - xq_cur) / (x_high - x_low)
#         yq[xqi_cur] = xqpi_cur * y[xi] + (1 - xqpi_cur) * y[xi + 1]
#     end
# end

function backward_iteration(Va, Pi, a_grid, z, r, beta, eis)
    # Step 1: discounting and expectations
    Wa = (beta .* Pi) * Va
    # Step 2: solving for asset policy using the first-order condition
    c_endog = Wa .^ (-eis)
    coh = z .+ (1 .+ float(r)) .* a_grid'

    a = similar(coh, eltype(coh))
    for s in 1:length(z)
        interp_knots = a_grid .+ c_endog[s, :]  # Points on function that we want to draw linear splines between
        #new_knots = Interpolations.deduplicate_knots!(interp_knots; move_knots = true) # Gets rid of duplicate points (by shifting duplicates by some negligible value), then sort in increasing order
        interp_final = LinearInterpolation(interp_knots, a_grid, extrapolation_bc=Line()) # Finally does interpolation on a_grid and c_endog 
        a[s, :] = interp_final(coh[s, :])
    end

    # Step 3: enforcing the borrowing constraint and backing out consumption
    a = max.(a, a_grid[1])
    c = coh - a

    # Step 4: using the envelope condition to recover the derivative of the value function
    Va = (1 .+ float(r)) .* (c .^ (-1 / eis))
    return Va, a, c
end


function policy_ss(Pi, a_grid, z, r, beta, eis, tol=1E-9)
    # Initial guess for Va: assume consumption 5% of cash-on-hand, then get Va from envelope condition
    # Require initial guess to be non-negative 
    a_grid_non_zero = max.(a_grid, 0)
    coh = z .+ (1 + r) .*  a_grid_non_zero'
    c = 0.05 .* coh
    Va = (1 + r) .* c .^ (-1 / eis)
    
    # Iterate until maximum distance between two iterations falls below tol, fail-safe max of 10,000 iterations
    a_old = similar(coh, eltype(coh))
    for it in 1:10_000
        Va, a, c = backward_iteration(Va, Pi, a_grid, z, r, beta, eis)
        
        # After iteration 0, can compare new policy function to old one
        if it > 1 && maximum(abs.(a .- a_old)) < tol
            return Va, a, c
        end
        copyto!(a_old, a)
    end
    
    error("Did not converge within the maximum number of iterations.")
end


"""Part 3: forward iteration for distribution"""

function get_lottery_helper(a, a_grid)
    # Step 1: find the i such that a' lies between gridpoints a_i and a_(i+1)
    a_i = searchsortedfirst(a_grid, a) - 1 # a_i is going to be +1 of what it would be in Python
    
    # Step 2: obtain lottery probabilities pi
    if a_i == 0
        a_pi = (a_grid[a_i + 1] - a) / (a_grid[a_i + 1] - a_grid[end])
    else
        a_pi = (a_grid[a_i + 1] - a) / (a_grid[a_i + 1] - a_grid[a_i])
    end
    
    return a_i, a_pi
end


function get_lottery(a, a_grid)
    a_i = similar(a, eltype(a))
    a_pi = similar(a, eltype(a))
    for row in 1:size(a, 1) 
        for column in 1:size(a, 2)
            a_i_helper, a_pi_helper = get_lottery_helper(a[row, column], a_grid)
            a_i[row, column] = a_i_helper
            a_pi[row, column] = a_pi_helper
        end
    end
    return Int.(a_i), a_pi
end


function forward_policy(D, a_i, a_pi)
    Dend = zeros(size(D))
    for e in 1:size(a_i, 1)
        for a in 1:size(a_i, 2)
            # Send pi(e,a) of the mass to gridpoint i(e,a)
            if a_i[e, a] == 0
                Dend[e, end] += a_pi[e, a] * D[e, a]
            else
                Dend[e, a_i[e, a]] += a_pi[e, a] * D[e, a]
            end
            
            # Send 1-pi(e,a) of the mass to gridpoint i(e,a)+1
            Dend[e, a_i[e, a] + 1] += (1 - a_pi[e, a]) * D[e, a]
        end
    end
    
    return Dend
end


function forward_iteration(D, Pi, a_i, a_pi)
    Dend = forward_policy(D, a_i, a_pi)
    return transpose(Pi) * Dend
end


function distribution_ss(Pi, a, a_grid, tol=1E-10)
    a_i, a_pi = get_lottery(a, a_grid)
    
    # As initial D, use stationary distribution for e, plus uniform over a
    pi = stationary_markov(Pi)
    D = (pi / length(a_grid)) .* ones(size(a))
    
    # Now iterate until convergence to acceptable threshold
    for _ in 1:10000
        D_new = forward_iteration(D, Pi, a_i, a_pi)
        if maximum(abs.(D_new .- D)) < tol
            return D_new
        end
        D = D_new
    end
    
    error("Did not converge within the maximum number of iterations.")
end


"""Part 4: solving for steady state, including aggregates"""

function steady_state(Pi, a_grid, z, r, beta, eis)
    Va, a, c = policy_ss(Pi, a_grid, z, r, beta, eis)
    a_i, a_pi = get_lottery(a, a_grid)
    D = distribution_ss(Pi, a, a_grid)
    
    return Dict("D" => D, "Va" => Va, 
                "a" => a, "c" => c, "a_i" => a_i, "a_pi" => a_pi,
                "A" => dot(a, D), "C" => dot(c, D),
                "Pi" => Pi, "a_grid" => a_grid, "z" => z, "r" => r, "beta" => beta, "eis" => eis)
end


"""Part 5: expectation iterations"""

function expectation_policy(Xend, a_i, a_pi)
    X = zeros(size(Xend))
    for e in 1:size(a_i, 1)
        for a in 1:size(a_i, 2)
            if a_i[e, a] == 0
                X[e, a] = a_pi[e, a] * Xend[e, end] + (1 - a_pi[e, a]) * Xend[e, a_i[e, a] + 1]
            else
                # Expectation is pi(e,a)*Xend(e,i(e,a)) + (1-pi(e,a))*Xend(e,i(e,a)+1)
                X[e, a] = a_pi[e, a] * Xend[e, a_i[e, a]] + (1 - a_pi[e, a]) * Xend[e, a_i[e, a] + 1]
            end
        end
    end

    return X
end


function expectation_iteration(X, Pi, a_i, a_pi)
    Xend = Pi * X
    return expectation_policy(Xend, a_i, a_pi)
end


function expectation_functions(X, Pi, a_i, a_pi, T)
    # Set up array of curlyEs and fill in first row with base case
    curlyE = Array{Float64}(undef, T, size(X, 1), size(X, 2))
    curlyE[1, :, :] = X
    
    # Recursively apply the law of iterated expectations
    for j in 2:T
        curlyE[j, :, :] = expectation_iteration(curlyE[j-1, :, :], Pi, a_i, a_pi)
    end
    
    return curlyE
end