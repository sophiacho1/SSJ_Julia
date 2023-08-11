
""" File contains the initialization of the DAG representation of the HANK model in addition to helper functions """

include("block.jl")
include("het.jl")
using OrderedCollections

struct DAG_Rep

    blocks::Array
    ins_blocks::Dict # A dictionary mapping outputs of a block to another block(s) that uses said outputs as inputs
    adj_list::Dict # An adjacency list representation of the DAG
    topsorted::Array
    jacobians::Array # List of Jacobians in DAG

    function DAG_Rep(blocks)
        ins_blocks = make_in_map(blocks)
        adj_list = make_adj_list(blocks, ins_blocks)
        topsorted = topsort(blocks, adj_list)
        jacobians = one_step_jacobians(blocks)
        return new(blocks, ins_blocks, adj_list, topsorted, jacobians)
    end
end



"+=
    Takes in a set of blocks and returns a dictionary where the keys represent
    inputs (r, π, etc.) and the values are the indices of the blocks that take
    these as inputs in the DAG representation of the model economy.
+="

function make_in_map(blocks)
    ins_blocks = Dict() # input -> blocks that use this as an input
    for (index, block) in enumerate(blocks)
        for input in keys(block.ins)
            if haskey(ins_blocks, input)
                push!(ins_blocks[input], index)
            else
                ins_blocks[input] = Set(index)
            end
        end
    end
    return ins_blocks
end



"+=
    There exists an edge (u, v) iff one of u's output is an input to v.
    Creates an adjacency list representation of the DAG.
=+"

function make_adj_list(blocks, ins_blocks)
    adj_list = Dict()
    for (index, block) in enumerate(blocks)
        adj_list[index] = Set()
        for output in keys(block.outs)
            if haskey(ins_blocks, output)
                for block_index in ins_blocks[output]
                    push!(adj_list[index], block_index)
                end
            end
        end
    end
    return adj_list
end



"+=
    Returns a topologically sorted ordering of the blocks. Do not
    modify the original!
+="

function topsort(blocks, adj_list)

    topsorted = []
    visited = Set()

    function recur_dfs(start)
        for neighbor in adj_list[start]
            if !(neighbor in visited)
                push!(visited, neighbor)
                recur_dfs(neighbor)
            end
        end
        push!(topsorted, start)
    end

    for index in 1:length(blocks)
        if !(index in visited)
            push!(visited, index)
            recur_dfs(index)
        end
    end
    
    topsorted = reverse(topsorted)

    for (i, block_index) in enumerate(topsorted)
        topsorted[i] = blocks[block_index]
    end

    return topsorted
end



"+= 
    Propagates through DAG to return steady state values for all inputs and outputs in the DAG (up to the HA block)
+="

function DAG_get_ss(economy::DAG_Rep, stop::String, hh_calibration)
    # Dictionary to collect exogenous variables at ss
    DAG_ss = Dict()
    hh_ss_all_vals = Dict()

    for block in economy.topsorted

        """ Get steady state of HA block """

        if block.name == "HA"
            DAG_ss["A"], DAG_ss["C"], hh_ss_all_vals = solve_ss(block, hh_calibration)
        end

        """ Get steady state for simple blocks """

        if block.name == stop
            break
        end
        try # Run function on block
            f_block(block)
        catch # The function is the identity function
        end

        for out in keys(block.outs)
            DAG_ss[out] = block.outs[out]
            try
                for children in c_hank.ins_blocks[out]
                    c_hank.blocks[children].ins[out] = block.outs[out]
                end
            catch # No block uses this output as an input
            end
        end
    end
    return DAG_ss, hh_ss_all_vals
end



"+=
    Given a DAG representation of the graph, return a dictionary of one-step Jacobians we need
    The dictionary is structured as follows:
    Keys are (o, i) represented J^{0, i}; values are nothing
+="

function one_step_jacobians(blocks) # Applies only to simple blocks (hh_block doesn't do Jacobian calculation when defined)
    all_Js = []
    for block in blocks
         # For all simple blocks, just append all of their Jacobians together
        for jacobian in block.jacobians
            push!(all_Js, jacobian)
        end
    end
    return all_Js
end



"+= 
    Forward Accumulation through DAG:
    - fw_acc finds the paths through which a variable of choice (svar) affects goods or asset clearing conditions (clearing_type)
+="

function fw_acc(economy::DAG_Rep, svar, clearing_type)

    # DAG; no worries about cyclicality
    paths = []

    function dfs(prev_input, start, clearing_type, path)
        if start.name == "Clearing"
            if clearing_type == "goods" # Specifies goods or asset market clearing 
                push!(path, "∂Hg/∂$prev_input")
            else
                push!(path, "∂Ha/∂$prev_input")
            end
            push!(paths, reverse(path))
            pop!(path)
        else
            # Recursively works through DAG to find path of the outputs affected by svar
            for (output, _) in start.outs
                push!(path, "∂$output/∂$prev_input")
                for next_idx in economy.ins_blocks[output]
                    dfs(output, c_hank.topsorted[next_idx], clearing_type, path)
                end
                pop!(path)
            end
        end
    end

    for start in economy.topsorted
        if haskey(start.ins, svar)
            dfs(svar, start, clearing_type, [])
        end # We don't care about the case where the block does not take svar as an input
    end
    return paths
end



"+= 
    Chaining Jacobians:
    - Constructs the relevant market clearing Jacobian w.r.t. to input (svar) stored in 'lmap.' Also creates symbolic representation of chained 
    Jacobians (stored in 'lmap_symbolic')
+="

function construct(paths, all_Jacobians, end_T)
    lmap = zeros(end_T, end_T)
    lmap_symbolic = ""
    for path in paths
        not_found = false
        current_chain = I
        current_symbolic = ""
        for element in path
            if haskey(all_Jacobians, element)
                current_chain = current_chain * all_Jacobians[element]
                current_symbolic = current_symbolic * element * " * "
            else
                not_found = true
                break
            end
        end
        if !not_found
            lmap = lmap + current_chain
            lmap_symbolic = lmap_symbolic * current_symbolic[1:end-3] * " + "
        end
    end
    return lmap, lmap_symbolic[1:end-3]
end