# pacSTL: PAC-Bounded Signal Temporal Logic from Data-Driven Reachability Analysis
This repository accompanies the paper subitted to L-CSS: "pacSTL: PAC-Bounded Signal Temporal Logic from Data-Driven Reachability Analysis".

# Parameters
## Vessel Reachable Sets: 

We consider the five dimensions relevant to the collision-avoidance specification: the vessel’s surface position p =[px, py], orientation ψ, and velocity v = [vx, vy]. We define a uniform random variable over the set of initial states, defined as the intervals below.

| Parameter | Value |
|-----------|------|
| px(0)     | [0.3, 0.5]     | 
| py(0)     | [-0.1, 0.1]    | 
| ψ(0)      | [-0.092, 0.0111] |
| vx(0)     | [0.092, 0]    | 
| vy(0)     | [-0.079, 0]   | 

Additionally, we define the set of inputs as the set of constant functions τ (t) = τ ∀t ∈ {t0, t1, t2, ..., tT } whose values are in the following interval:

τ_low = [0.7, -0.1, 0, 0, 0], τ_high = [1.2, 0.1, 0, 0, 0]

## Vessel Navigation Monitoring:

| Parameter | Value|
|-----------|------|
| vmax   |         | 
| rc     |         |  
| th     |         |


## Quadrotor Reachable Sets:
The quadrotor model consists of pn, pe, and h (the y-axis, "north", the x-axis, "east", and the z-axis, "height"), ϕ, θ, and ψ (pitch, roll, and yaw angles respectively), as well as their first derivatives (pn', pe', h', ϕ', θ', and ψ'), resulting in the state vector: [pn, pe, h, ϕ, θ, ψ, pn', pe', h', ϕ', θ', ψ']. We define a normal random variable over the set of initial states, parameterized as the following distribution. 

| Parameter | Normal Distribution Parameters |
|-----------|------|
| pn(0)    | μ = 0 , σ = 0.15 | 
| pe(0)    | μ = 0 , σ = 0.15 | 
| h(0)     | μ = 0 , σ = 0.15 |
| ϕ(0)     | μ = 0 , σ = 0.02 | 
| θ(0)     | μ = 0 , σ = 0.02 | 
| ψ(0)     | μ = 0 , σ = 0.02 | 
| pn'(0)    | μ = 0 , σ = 0.15 | 
| pe'(0)    | μ = 0 , σ = 0.15 | 
| h'(0)     | μ = 0 , σ = 0.15 |
| ϕ'(0)     | 0 | 
| θ'(0)     | 0 | 
| ψ'(0)     | 0 | 

Additionally, we add constant disturbance to ϕ'', θ'', and ψ'', to mimic unbounded real-world uncertainty, parameterized as the following set of functions:
ϕ''(t) = ϕ''(t) + (μ = 0 , σ = 0.02), θ''(t) = θ''(t) + (μ = 0 , σ = 0.02), and ψ''(t) = ψ''(t) + (μ = 0 , σ = 0.02).

## Quadrotor Verification: