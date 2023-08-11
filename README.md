### Sequence State Jacobian (SSJ) Solution Method in Julia
---
We follow [NBER's heterogeneous-agent macro workshop](https://github.com/shade-econ/nber-workshop-2023) to implement the Sequence Space Jacobian (SSJ) solution method in Julia, specifically to solve the one-asset HANK model.

Within the **one asset hank** folder, there are two folders (**fiscal policy** and **monetary policy**) that contain the files needed to solve the one-asset HANK model with fiscal and monetary policy, respectively.

Each of the **fiscal policy** and **monetary policy** folders contain the following files:
- **block.jl**: code for the initialization of simple blocks
  - struct Block
  - function f_block
  - function simple_solve_jacobian
- **graph.jl**: code to intialize and utilize the DAG representation of the model
  - struct DAG_Rep
  - function make_in_map
  - function make_adj_list
  - function topsort
  - function DAG_get_ss
  - function one_step_jacobians
  - function fw_acc
  - function construct
- **hank.jl**:
- **het.jl**:
- **sim_steady_state.jl**:
- **tutorial.ipynb**:
