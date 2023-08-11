
""" File contains initialization of simple blocks (e.g. Fiscal and Market Clearing Blocks) """

mutable struct Block
    ins
    outs
    f # Function this block is performing
    name 
    jacobians 
    function Block(ins, outs, f, ins_outs, name = "Not named")
        # Populate the block with the symbolic form of the block's Jacobian 
        jacobians = simple_solve_jacobian(ins_outs, outs)
        return new(ins, outs, f, name, jacobians)
    end
end

# Apply function to block
function f_block(block::Block)
    block.f(block)
end

# Gets relevant Jacobians for each simple block (empty dictionaries)
function simple_solve_jacobian(ins_outs, outs)
    jacobian_list = []
    for output in keys(outs) # Loop through outputs of block
        for dict_pair in ins_outs # Loop through all the dictionaries in ins_outs list
            if output in dict_pair["outs"] # If output is one of the outs in the ins_outs list
                for input in dict_pair["ins"] # For every associated input in that dictionary
                    push!(jacobian_list, Dict((output, input) => "∂$output/∂$input")) # Append to jacobian_list
                end
            end
        end
    end        
    return jacobian_list
end

# Show inputs and outputs of block
function Base.show(block::Block)
    println("Block named $name with inputs: ", block.ins, " and outputs: ", block.outs)
end