include("het.jl")
include("graph.jl")

# NOTE: Z is technically Y

""" I. Specifies the ins and outs of each block in the DAG"""

ins_outs = [
    Dict("ins" => ["r_ante"], "outs" => ["r"]), # no-val block
    Dict("ins" => ["r", "Z"], "outs" => ["C", "A"]), # ha block
    Dict("ins" => ["A", "C", "Z"], "outs" => ["asset_mkt", "goods_mkt"]), # goods market clearing block
]

""" II. Defining the no-val block """

dict_in_ex_post_rate = Dict{Any, Any}(
    "r_ante" => 0.01,
)
dict_out_ex_post_rate = Dict{Any, Any}(
    "r" => nothing
)

function f_ex_post_rate(block::Block)
    r_ante = block.ins["r_ante"]
    r = r_ante
    block.outs["r"] = r
end

# ex_post_rate_block = Block(dict_in_ex_post_rate, dict_out_ex_post_rate, f_ex_post_rate, ins_outs, "No-Val")

""" III. Defining the ha block """

hh_block = HH_Block(Dict{Any, Any}("r" => nothing, "Z" => 1.0), Dict{Any, Any}("C" => nothing), "HA", []) # Defining a placeholder for the heterogeneous agent block

""" IV. Defining the goods market clearing block """

dict_in_mkt_clearing_simple = Dict{Any, Any}(
    "A" => nothing,
    "C" => nothing,
    "Z" => 1.0
)
dict_out_mkt_clearing_simple = Dict{Any, Any}(
    "asset_mkt" => nothing,
    "goods_mkt" => nothing
)

function f_mkt_clearing_simple(block::Block)
    A, C, Z = block.ins["A"], block.ins["C"], block.ins["Z"]
    asset_mkt = A
    goods_mkt = C - Z
    block.outs["asset_mkt"] = asset_mkt
    block.outs["goods_mkt"] = goods_mkt
end

mkt_clearing_simple_block = Block(dict_in_mkt_clearing_simple, dict_out_mkt_clearing_simple, f_mkt_clearing_simple, ins_outs, "Clearing")

""" V. Creating the DAG """

mon_DAG =  DAG_Rep([hh_block, mkt_clearing_simple_block])

calibration = Dict("eis" => 0.5,     # EIS
                   "rho_e" => 0.92,  # Persistence of idiosyncratic productivity shocks
                   "sd_e" => 0.92,   # Standard deviation of idiosyncratic productivity shocks
                   "Z" => 1.0,       # Output
                   "r_ante" => 0.01, # target real interest rate
                   "min_a" => -1.0,  # Minimum asset level on the grid
                   "max_a" => 1_000, # Maximum asset level on the grid
                   "n_a" => 300,     # Number of asset grid points
                   "n_e" => 11,      # Number of productivity grid points
                   "beta" => 0.85)     

inputs_to_hh = Dict("eis" => calibration["eis"], 
                    "rho_e" => calibration["rho_e"],
                    "sd_e" => calibration["sd_e"],
                    "r" => calibration["r_ante"],
                    "min_a" => calibration["min_a"],
                    "max_a" => calibration["max_a"],
                    "n_a" => calibration["n_a"],
                    "n_e" => calibration["n_e"],
                    "beta" => calibration["beta"])

ss_vals, hh_ss_all_vals = DAG_get_ss(mon_DAG, "None", inputs_to_hh, "mon_beta")

# Ex-Post Block jacobian
# J_r_rante = I

# Get HA block jacobians 
end_T = 300
het_Js = ha_jacobian(hh_ss_all_vals, Dict("r" => Dict("r" => 1), "z" => Dict("z" => 1)), 1.0, 0.01, end_T)

