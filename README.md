# pacSTL: PAC-Bounded Signal Temporal Logic from Data-Driven Reachability Analysis
This repository accompanies the paper submitted to L-CSS: "pacSTL: PAC-Bounded Signal Temporal Logic from Data-Driven Reachability Analysis".

# Parameters

## Quadrotor 

The system dynamics according to the quadrotor model in Chapter 8.2 [1], which uses the parameters from [2]. The quadrotor model consists of $p_n$, $p_e$, and $h$ (the y-axis, "north", the x-axis, "east", and the z-axis, "height"), $\phi$, $\theta$, and $\psi$ (pitch, roll, and yaw angles respectively), as well as their first derivatives ($\dot{p}_n$, $\dot{p}_e$, $\dot{h}$, $\dot{\phi}$, $\dot{\theta}$, and $\dot{\psi}$), resulting in the state vector: $[p_n, p_e, h, \phi, \theta, \psi, \dot{p}_n, \dot{p}_e, \dot{h}, \dot{\phi}, \dot{\theta}, \dot{\psi}]$. We define a normal random variable over the set of initial states, parameterized as the following distribution. 

| Parameter | Normal Distribution Parameters |
|-----------|------|
| $p_n(0)$  | $\mu = 0, \sigma = 0.15$ | 
| $p_e(0)$  | $\mu = 0, \sigma = 0.15$ | 
| $h(0)$    | $\mu = 0, \sigma = 0.15$ |
| $\phi(0)$ | $\mu = 0, \sigma = 0.02$ | 
| $\theta(0)$| $\mu = 0, \sigma = 0.02$ | 
| $\psi(0)$ | $\mu = 0, \sigma = 0.02$ | 
| $\dot{p}_n(0)$ | $\mu = 0, \sigma = 0.15$ | 
| $\dot{p}_e(0)$ | $\mu = 0, \sigma = 0.15$ | 
| $\dot{h}(0)$ | $\mu = 0, \sigma = 0.15$ |
| $\dot{\phi}(0)$ | $0$ | 
| $\dot{\theta}(0)$| $0$ | 
| $\dot{\psi}(0)$ | $0$ | 

Additionally, we add constant disturbance to $\ddot{\phi}$, $\ddot{\theta}$, and $\ddot{\psi}$, to mimic unbounded real-world uncertainty, parameterized as the following set of functions:

$$\ddot{\phi}(t) = \ddot{\phi}(t) + N(\mu = 0, \sigma = 0.02)$$
$$\ddot{\theta}(t) = \ddot{\theta}(t) + N(\mu = 0, \sigma = 0.02)$$
$$\ddot{\psi}(t) = \ddot{\psi}(t) + N(\mu = 0, \sigma = 0.02)$$


## Vessel 

**Reachable sets of obstacle vessel**

The obstacle vessel is simulated with a six-degree-of-freedom (6-DOF) model (Eqs. (2.1) and (2.2), [3]). Let $\mathbf{v} = [u,v,w]^\top$, $\omega = [p,q,r]^\top$, and $S(\cdot)$ denote the skew-symmetric operator. For solids with uniform density, let the mass be calculated as $m = \rho L B T$ where $L$ is the length of the vessel, $B$ is the beam/width, and $T$ is the height/depth.
Then, we define the rigid-body Coriolis and centripetal terms as follows:

$$
\mathbf{C}_{\mathrm{RB}}(\nu) = \begin{bmatrix} \mathbf{0} & -m S(\omega) \\ - m S(\mathbf{v}) & - S(\mathbf{I}\omega) \end{bmatrix}, \quad \mathbf{I}\omega = [I_x p, I_y q, I_z r],
$$

where $\mathbf{I}$ and the moments of inertia $I_x, I_y, I_z$ are defined in \cite{gezer2025digital}. 
For added mass terms, this is partitioned into linear and rotational components,

$$
\mathbf{C}_{\mathrm{A}}(\nu) = \begin{bmatrix} \mathbf{0} & - S(\mathbf{M}_{\mathrm{A,lin}}\mathbf{v}) \\ - S(\mathbf{M}_{\mathrm{A,lin}}\mathbf{v}) & - S(\mathbf{M}_{\mathrm{A,rot}}\omega) \end{bmatrix}.
$$

Therefore, $\mathbf{C} = \mathbf{C}_{\mathrm{RB}}(\nu) + \mathbf{C}_{\mathrm{A}}(\nu)$.
Further, the added mass and effective mass is defined as

$$
\begin{aligned} \mathbf{M}_{RB} &= \text{diag}(m, m, m, I_x, I_y, I_z),\\ \mathbf{M}_{A} &= \begin{bmatrix}\mathbf{M}_{\mathrm{A,lin}} & \mathbf{0}_{3\times3} \\ \mathbf{0}_{3\times3} & \mathbf{M}_{\mathrm{A,rot}}\end{bmatrix}, \quad \mathbf{M} = \mathbf{M}_A + \mathbf{M}_{RB} \end{aligned}
$$

where $\mathbf{M}_{RB}$ is the rigid body mass and inertia. Based on these equations the necessary parameters are:

$$\rho = 1000.0, B = 0.440, L = 2.578, T =0.02$$
$$\mathbf{M} = diag(132.0, 144.0, 240.0, 1.9, 99.1, 100.76)$$
$$\mathbf{M}_{\mathrm{A,lin}} = diag(12.0, 24.0, 120.0), $$
$$\mathbf{M}_{\mathrm{A,rot}} = diag(0.17, 9.01, 9.16), $$
$$\mathbf{D} = diag(30.0, 30.0, 30.0, 0.425, 22.525, 22.90).$$

We consider the five dimensions relevant to the specification and thus reachable sets: the vessel’s surface position $p = [p_x, p_y]$, orientation $\psi$, and velocity $v = [v_x, v_y]$. We define a uniform random variable over the set of initial states, defined as the intervals below.

| Parameter | Value |
|-----------|------|
| $p_x(0)$  | $[0.3, 0.5]$  | 
| $p_y(0)$  | $[-0.1, 0.1]$ | 
| $\psi(0)$ | $[-0.092, 0.0111]$ |
| $v_x(0)$  | $[0.092, 0]$  | 
| $v_y(0)$  | $[-0.079, 0]$ | 

Additionally, we define the set of inputs as the set of constant functions $\tau(t) = \tau \quad \forall t \in \{t_0, t_1, t_2, \dots, t_T\}$ whose values are in the following interval:

$$\tau_{\text{low}} = [0.7, -0.1, 0, 0, 0], \quad \tau_{\text{high}} = [1.2, 0.1, 0, 0, 0]$$

**Vessel navigation**

The ego vessel is initialized at $p_x =  7.0, p_y = -1,0, \psi = 2.89, v_x = -0.18, v_y = 0.052$ and controlled by a waypoint tracking controller. The obstacle vessel uses the 6-DOF model and samples a random control input from $\tau$ every 5 time steps. Initially, the obstacle vessel is at $p_x = -4.0, p_y = 1.5, \psi = 5.81, v_x = 0.33, v_y = 0.0$. The fixed parameters of the specification $\Phi$ are:

| Parameter | Value|
|-----------|------|
| $v_{\text{max}}$ | 0.4| 
| $r_e$     | 1.0 |  
| $t_h$     | 20.0 |



# Installation 
<pre>git clone https://github.com/eadietri/pacSTL_LCSS.git
</pre>

<pre>python -m venv venv
source venv/bin/activate
</pre>

<pre>pip install -e .
</pre>

# Run Instructions

# References

[1] P.-J. Meyer, A. Devonport, and M. Arcak, Interval Reachability Analysis: Bounding Trajectories of Uncertain Systems with Boxes for Control and Verification, 2021.

[2] F. Immler, M. Althoff, X. Chen, C. Fan, G. Frehse, N. Kochdumper, Y. Li, S. Mitra, M.S. Tomar, and M. Zamani, "ARCH-COMP18 category report: continuous and hybrid systems with nonlinear dynamics," In: Proceedings of the 5th International Workshop on Applied Verification for Continuous and Hybrid Systems, 2018.

[3] T. I. Fossen, Handbook of marine craft hydrodynamics and motion control. John Wiley & Sons, 2011.

[4]  E. C. Gezer, M. K. I. Moreau, A. S. Høgden, D. T. Nguyen, R. Skjetne, and A. Sørensen, “Digital-physical testbed for ship autonomy studies in the Marine Cybernetics Laboratory basin,” arXiv:2505.06787, 2025.





