# pacSTL: PAC-Bounded Signal Temporal Logic from Data-Driven Reachability Analysis
This repository accompanies the paper subitted to L-CSS: "pacSTL: PAC-Bounded Signal Temporal Logic from Data-Driven Reachability Analysis".

# Parameters
Vessel Reachable Sets: 

We consider the five dimensions relevant to the collision-avoidance specification: the vessel’s surface position p =[px, py], orientation ψ, and velocity v = [vx, vy]. We define a uniform random variable over the set of initial states, defined as the intervals below.

| Parameter | Value |
|-----------|------|
| px(0)     | [0.3, 0.5]     | 
| py(0)     | [-0.1, 0.1]    | 
| ψ(0)      | [-0.092, 0.0111] |
| vx(0)     | [0.092, 0]    | 
| vy(0)     | [-0.079, 0]   | 

Additionally, we define the set of inputs as the set of constant functions τ (t) = τ ∀t ∈ {t0, t1, t2, ..., tT } whose values are in the following interval:

τ_low = [0.7, -0.1, 0, 0, 0]
τ_high = [1.2, 0.1, 0, 0, 0]

Vessel Navigation:

| Parameter | Value|
|-----------|------|
| vmax   |         | 
| rc     |         |  
| th     |         |

