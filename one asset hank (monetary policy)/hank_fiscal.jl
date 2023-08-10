include("het.jl")
include("graph.jl")

""" I. Specifies the ins and outs of each block in the DAG"""

ins_outs = [
    Dict("ins" => ["r", "G", "B", "Y"], "outs" => ["Z"]), # Fiscal block
    Dict("ins" => ["Z", "r"], "outs" => ["A"]), # HA block
    Dict("ins" => ["A", "B"], "outs" => ["asset_mkt"]), # Asset market clearing
    Dict("ins" => ["Y", "C", "G"], "outs" => ["goods_mkt"]), # Goods market clearing
]



""" II. Defining the Fiscal Block, including equations that transform inputs to output (Z) """

dict_in_fiscal = Dict{Any, Any}(
    "B" => 0.8,
    "r" => 0.03,
    "G" => 0.2,
    "Y" => 1.0
)
dict_out_fiscal = Dict{Any, Any}(
    "Z" => nothing
)

"+=
fiscal_function: inputs -> outputs
=+"

function f_fiscal(block::Block)
    B, r, G, Y = block.ins["B"], block.ins["r"], block.ins["G"], block.ins["Y"]
    T = G + r * B # B(-1) in the original notebook
    Z = Y - T
    block.outs["Z"] = Z
end

fiscal_block = Block(dict_in_fiscal, dict_out_fiscal, f_fiscal, ins_outs, "Fiscal") # Defining the fiscal block



""" III. Defining the Market Clearing Block """

function f_mkt_clearing(block::Block)
    A, B, Y, C, G = block.ins["A"], block.ins["B"], block.ins["Y"], block.ins["C"], block.ins["G"]
    asset_mkt = A - B
    goods_mkt = Y - C - G
    block.outs["asset_mkt"] = asset_mkt
    block.outs["goods_mkt"] = goods_mkt
end

dict_in_mkt = Dict{Any, Any}(
    "A" => nothing,
    "B" => 0.8,
    "Y" => 1.0,
    "C" => nothing,
    "G" => 0.2
)
dict_out_mkt = Dict{Any, Any}(
    "asset_mkt" => nothing,
    "goods_mkt" => nothing
)
mkt_clearing_block = Block(dict_in_mkt, dict_out_mkt, f_mkt_clearing, ins_outs, "Clearing")



""" IV. Defining the Household (HA) Block """

hh_block = HH_Block(Dict{Any, Any}("Z" => nothing, "r" => nothing), Dict{Any, Any}("A" => nothing, "C" => nothing), "HA", []) # Defining a placeholder for the heterogeneous agent block



""" V. Creating the DAG 
1. Order of blocks in the DAG initialization statement doesn't matter --> topological sort is applied when DAG is created
2. HA block is then solved in the steady state using the DAG_ss function """

# Defining the DAG
c_hank = DAG_Rep([fiscal_block, hh_block, mkt_clearing_block])

# Inputs to entire DAG
calibration = Dict("eis" => 0.5,  # Elasticity of intertemporal substitution
                   "rho_e" => 0.9,  # Persistence of idiosyncratic productivity shocks
                   "sd_e" => 0.92,  # Standard deviation of idiosyncratic productivity shocks
                   "G" => 0.2,  # Government spending
                   "B" => 0.8,  # Government debt
                   "Y" => 1.,  # Output
                   "min_a" => 0.,  # Minimum asset level on the grid
                   "max_a" => 1_000,  # Maximum asset level on the grid
                   "n_a" => 200,  # Number of asset grid points
                   "n_e" => 10,  # Number of productivity grid points
                   "r" => 0.03, # Interest rate
                   "beta" => 0.85) # Discount factor

# Specific inputs to the HA block
inputs_to_hh = Dict("eis" => calibration["eis"], 
                    "rho_e" => calibration["rho_e"],
                    "sd_e" => calibration["sd_e"],
                    "n_e" => calibration["n_e"],
                    "min_a" => calibration["min_a"],
                    "max_a" => calibration["max_a"],
                    "n_a" => calibration["n_a"],
                    "r" => calibration["r"],
                    "beta" => calibration["beta"],
                    "B" => calibration["B"])

# Print steady state exogenous values from DAG
##--> ss_vals returns steady-state outputs (Z, A, C, asset_mkt, goods_mkt); hh_ss_all_vals returns steady state outputs of the backpropogation function 

ss_vals, hh_ss_all_vals = DAG_get_ss(c_hank, "None", inputs_to_hh, "fisc_beta")