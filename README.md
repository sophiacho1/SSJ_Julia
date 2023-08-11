### Sequence State Jacobian (SSJ) Solution Method in Julia
---
We follow [NBER's heterogeneous-agent macro workshop](https://github.com/shade-econ/nber-workshop-2023) to implement the Sequence Space Jacobian (SSJ) solution method in Julia, specifically to solve the one-asset HANK model.

Within the **one asset hank** folder, there are two folders (**fiscal policy** and **monetary policy**) that contain the files needed to solve the one-asset HANK model with fiscal and monetary policy, respectively.

Each of the **fiscal policy** and **monetary policy** folders contain the following files:
- **block.jl**: code to intialize simple blocks
  - struct Block
  - function f_block
  - function simple_solve_jacobian
- **sim_steady_state**: translation of [sim_steady_state.py](https://github.com/shade-econ/nber-workshop-2023/blob/main/Lectures/sim_steady_state.py) 
- **het.jl**
  - struct HH_Block
  - function mk_grids_and_income
  - function solve_ss
  - function J_from_F
  - function get_shocked_inputs
  - function get_shocked_inputs_no_compound
  - function unpack_backiter
  - function step1_backward
  - function ha_jacobian
  - function policy_impulse
  - function distribution_impulse
  - function household_impulse
  - function impulse_map
- **graph.jl**: code to intialize and utilize the DAG representation of the model
  - struct DAG_Rep
  - function make_in_map
  - function make_adj_list
  - function topsort
  - function DAG_get_ss
  - function one_step_jacobians
  - function fw_acc
  - function construct
- **hank.jl**: code to run the model
- **tutorial.ipynb**: notebook to give step-by-step instruction on how to use our package (mirrors Tutorials 1 and 2 of the workshop)
