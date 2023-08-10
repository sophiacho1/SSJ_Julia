""" File contains helper functions for heterogeneous agent block """

include("sim_steady_state.jl")
using Roots


mutable struct HH_Block
  ins::Dict
  outs::Dict
  name::String
  jacobians::Array
  all_ins::Dict
  function HH_Block(ins, outs, name, jacobians)
    # Creates dictionary with all relevant inputs to the HA block 
    # (this includes some parameters from the calibration dictionary necessary for ss calculations)
    all_ins = Dict() 
    return new(ins, outs, name, jacobians, all_ins)
  end
end


# "+= Makes asset and income grids, and calculates after-tax income for HA block"
 function mk_grids_and_income(rho_e, sd_e, n_e, min_a, max_a, n_a, Z)
   # 1. Calculate asset grid and productivity grid respectively 
   a_grid = discretize_assets(min_a, max_a, n_a)

   # e_grid only used to calculate post-tax income z
   e_grid, _, Pi = discretize_income(rho_e, sd_e, n_e)

   # 2. Calculate after-tax income
   z = Z .* e_grid

   return a_grid, z, e_grid, Pi
 end


"+=
Takes in an instance of the household block as well as additional parameters (ins) needed to solve for the steady state
Returns a household block as well as general equilibrium values
+="
function solve_ss(hh::HH_Block, ins, beta_pol_type)
  # Create block instance
  hh.all_ins = merge(hh.ins, ins)
  hh.ins["r"] = hh.all_ins["r"]

  a_grid, z, e_grid, Pi = mk_grids_and_income(hh.all_ins["rho_e"], hh.all_ins["sd_e"], hh.all_ins["n_e"], hh.all_ins["min_a"], hh.all_ins["max_a"], hh.all_ins["n_a"], hh.all_ins["Z"])

  ss = steady_state(Pi, a_grid, z, hh.all_ins["r"], hh.all_ins["beta"], hh.all_ins["eis"])

  # Defining general equilibrium beta
  beta_ge = 0.0

  # Distinguish between asset/goods market clearing beta for fisc policy and asset/goods market clearing beta for mon policy
  
  try
    if beta_pol_type == "fisc_beta"
      function objective_function_fisc(beta)
        result_fisc = steady_state(ss["Pi"], ss["a_grid"], z, ss["r"], beta, ss["eis"])["A"] - hh.all_ins["B"] 
        return result_fisc
      end

      beta_ge = find_zero(objective_function_fisc, (0.75, 0.9), Bisection()) # Find the root using find_zero with the Bisection method
    end

    if beta_pol_type == "mon_beta"
      function objective_function_mon(beta)
        result_mon = steady_state(ss["Pi"], ss["a_grid"], z, ss["r"], beta, ss["eis"])["C"] - hh.all_ins["Z"] 
        return result_mon
      end
    end
      beta_ge = find_zero(objective_function_mon, (0.75, 0.9), Bisection()) # Find the root using find_zero with the Bisection method

  catch
      nothing
  end

  #beta_ge = hh.all_ins["beta"]
  # Get steady state of hh
  hh_ge = steady_state(Pi, a_grid, z, hh.all_ins["r"], beta_ge, hh.all_ins["eis"])
  hh_ge["e_grid"] = e_grid
  
  hh.outs = Dict("A" => hh_ge["A"], "C" => hh_ge["C"])
                    
  return hh.outs["A"], hh.outs["C"], hh_ge
end

"+=
  Fake News Algorithm: Constructing HA block Jacobians (w.r.t. Z and r)
  (Labeled I-IV below)
+="



"+=
  I. Construct F matrix after doing period t=1 fake-out
+="
# FAKE NEWS ALGORITHM
function J_from_F(F)
  J = copy(F)
  for t in 1:size(F, 1)-1
    J[2:end, t+1] += J[1:end-1, t]
  end
  return J
end



"+=
  II. 
  - 'get_shocked_inputs' takes shock to r or Z and returns new values of r or Z. This function compounds (i.e. shocks to r also affect Z directly)
  - 'get_shocked_inputs_no_compound' is similar to the former. This function doesn't compound (i.e. treats shocks to r separately from shocks to Z)
    --> Note that calculating the A_r and C_r Jacobians assume there is no compound effect (i.e. r directly affects A and C, not through the channel of Z)
    --> Inputs (in order): steady_state dictionary, shock, aggr output, current bond issuance, prior bond issuance, govt spending
+="

# function get_shocked_inputs(ss, shock, Y, B, B_prev, G, h)
#   shocked_inputs = Dict()
#   for key in keys(shock) # Has only one key
#     if key == "r"
#       shocked_inputs[key] = ss[key] + h * shock[key] # Change in interest rate
#       shocked_inputs["z"] = (Y - (1 + shocked_inputs["r"]) * B_prev + B - G) .* ss["e_grid"] # Change in income
#     else
#       shocked_inputs[key] = (Y + h - (1 + ss["r"]) * B_prev + B - G) .* ss["e_grid"] # Change in z is entered proportionally to maintain income risk
#     end
#   end
#   return shocked_inputs
# end


function get_shocked_inputs_no_compound(ss, shock, z, r, h)
  shocked_inputs = Dict()
  for key in keys(shock) #
    if key == "r"
      shocked_inputs[key] = ss[key] .+ (h .* shock[key]) # Change in interest rate/Z
    else #Hard coded Z from calibration
      shocked_inputs[key] = (1.0 .+ (h .* shock[key])) .* ss["e_grid"]
    end
  end
  return shocked_inputs
end



"+=
  III. 
  - 'unpack_backiter' is a helper function to step1_backward. Extracts key value from inputs to steady_state calculation to use in back iter
  - 'step1_backwards' applies the fake-out (gets how the distribution changes when there is anticipation of a fake shock at t=1, then immediately revert to ss values)
+="

function unpack_backiter(ss_inputs, shocked_dict)
  backiter_inputs = []
  for key in ["Va", "Pi", "a_grid", "z", "r", "beta", "eis"]
    if haskey(shocked_dict, key)
      push!(backiter_inputs, shocked_dict[key])
    else
      push!(backiter_inputs, ss_inputs[key])
    end
  end
  return backiter_inputs
end


function step1_backward(ss, shock, z, r, h, T)
  # Preliminaries: D_1 with no shock, ss inputs to backward_iteration
  D1_noshock = forward_iteration(ss["D"], ss["Pi"], ss["a_i"], ss["a_pi"])
  ss_inputs = Dict(k => ss[k] for k in ("Va", "Pi", "a_grid", "z", "r", "beta", "eis"))
    
  # Allocate space for results
  curlyY = Dict("A" => zeros(T), "C" => zeros(T))
  curlyD = zeros(T, size(ss["D"], 1), size(ss["D"], 2))
    
  # Backward iterate
  Va = ss_inputs["Va"] # Need to initialize Va in Julia
  for s in 0:T-1
    if s == 0
      # At horizon of s=0, 'shock' actually hits, override ss_inputs with shock
      shocked_inputs = get_shocked_inputs_no_compound(ss, shock, z, r, h)
      back_iter_inputs = unpack_backiter(ss_inputs, shocked_inputs)
      Va, a, c = backward_iteration(back_iter_inputs...)
    else
      # Now the only effect is anticipation, so it's just Va being different
      Va, a, c = backward_iteration(Va, 
                                    ss_inputs["Pi"], 
                                    ss_inputs["a_grid"], 
                                    ss_inputs["z"], 
                                    ss_inputs["r"], 
                                    ss_inputs["beta"], 
                                    ss_inputs["eis"])
    end
        
    # Aggregate effects on A and C (first row in the F-matrix)
    curlyY["A"][s+1] = dot(ss["D"], a - ss["a"]) ./ h
    curlyY["C"][s+1] = dot(ss["D"], c - ss["c"]) ./ h
        
    # What is effect on one-period-ahead distribution?
    a_i_shocked, a_pi_shocked = get_lottery(a, ss["a_grid"])
    curlyD[s+1, :, :] = (forward_iteration(ss["D"], ss["Pi"], a_i_shocked, a_pi_shocked) - D1_noshock) ./ h
  end
    
  return curlyY, curlyD
end
  


"+=
  IV. Creates Jacobian using fake news algorithm (J_A_Z, J_A_r, J_C_Z, J_C_r)
  - d_shocked_var = deviations from time 0 to T-1 of the shocked variable from its steady state value
+="

function ha_jacobian(ss, shocks, z, r, T)
  # Step 1: for all shocks i, allocate to curlyY[o][i] and curlyD[i]
  curlyY = Dict("A" => Dict(), "C" => Dict())
  curlyD = Dict()
  for (i, shock) in shocks
      curlyYi, curlyD[i] = step1_backward(ss, shock, z, r, 1E-4, T)
      curlyY["A"][i], curlyY["C"][i] = curlyYi["A"], curlyYi["C"]
  end
  
  # Step 2: for all outputs o of interest (here A and C)
  curlyE = Dict()
  for o in ("A", "C")
      curlyE[o] = expectation_functions(ss[lowercase(o)], ss["Pi"], ss["a_i"], ss["a_pi"], T-1)
  end
  
  # Steps 3 and 4: build fake news matrices, convert to Jacobians
  Js = Dict("A" => Dict(), "C" => Dict())
  for o in keys(Js)
      for i in keys(shocks)
          F = zeros(T, T)
          F[1, :] = curlyY[o][i]
          F[2:end, 1:end] = reshape(curlyE[o], T-1, :) * reshape(curlyD[i], T, :)'
          Js[o][i] = J_from_F(F)
      end
  end
  
  return Js
end


"+=
  Helpers for impulse response calculations
+="

"+=
ss: steady state dictionary
shocks: dictionary; d_shock variable; added to steady state values
T: calculate household response up to T
+="

function policy_impulse(ss, shocks, T)
    # Check that all values in "shocks" have first dimension T
    @assert all(size(x,1) == T for x in values(shocks))

    # Extract inputs to backward_iteration function from ss
    inputs = Dict(k => ss[k] for k in ("Va", "Pi", "a_grid", "z", "r", "beta", "eis"))

    # Create a T*nS*nA array to store each output of backward iteration
    Va = Array{Float64}(undef, T, size(ss["Va"], 1), size(ss["Va"], 2))
    a = Array{Float64}(undef, T, size(ss["Va"], 1), size(ss["Va"], 2))
    c = Array{Float64}(undef, T, size(ss["Va"], 1), size(ss["Va"], 2))

    for t in reverse(1:T)
        # Add this period's perturbation to parameters that are shocked
        for k in ("z", "r")
            if k in keys(shocks)
              if k == "r"
                inputs[k] = ss[k] .+ shocks[k][t, :]
              else
                inputs[k] = (1 .+ shocks[k][t, :]) .* ss[k] # Change in z is entered proportionally to maintain income risk
              end
            end
        end
        # println("Inputs:")
        # println(inputs["r"])
        # println("Shocks:")
        # println(shocks["r"][t, :])

        Va[t, :, :], a[t, :, :], c[t, :, :] = backward_iteration(inputs["Va"], inputs["Pi"], inputs["a_grid"], inputs["z"], 
                                                                 inputs["r"], inputs["beta"], inputs["eis"])

        # Use this Va for the next iteration
        inputs["Va"] = Va[t, :, :]
        #println(t)
    end

    return Va, a, c
end



""" Finds how distribution changes w.r.t. to shock """

function distribution_impulse(ss, a, T)
  @assert size(a, 1) == T
  D = similar(a)
  D[1, :, :] = ss["D"]
  
  for t in 1:T-1
      a_i, a_pi = get_lottery(a[t, :, :], ss["a_grid"])
      D[t+1, :, :] = forward_iteration(D[t, :, :], ss["Pi"], a_i, a_pi)
  end
  
  return D
end



""" Finds how household block outputs change w.r.t. to shock """

function household_impulse(ss, shocks, T)
  Va, a, c = policy_impulse(ss, shocks, T)
  D = distribution_impulse(ss, a, T)
  return Dict("D" => D, "Va" => Va, "a" => a, "c" => c, # time-varying stuff
              "A" => vec(sum(a.*D, dims=(2,3))), "C" => vec(sum(c.*D, dims=(2,3)))) # aggregate everything else quickly
end



""" Return asset_market and goods_market errors """

function impulse_map(rs, Zs, ss, T)
  r_dict = rs .- ss["r"]
  z_dict = zeros(T, 11)
  impulse = household_impulse(ss, Dict("r" => r_dict, "z" => z_dict), T)
  return impulse["C"] - Zs, impulse
end