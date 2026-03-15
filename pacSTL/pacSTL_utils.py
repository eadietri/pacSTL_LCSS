import copy

import numpy as np

class EllipsoidalSignalTemporalLogic:
    """
    This class implements Ellipsoidal Signal Temporal Logic (eSTL) which extends I-STL to incorporate bounded uncertainty in signal values
    and predicate functions using ellipsoids. eSTL syntax is the same as I-STL, but the calulcation of intervals is done using
    ellipsoidal reachable sets. Here, we define the quantitative sementics of eSTL.

    Attributes:
        phi_low : lower bound of the eSTL formula
        phi_high : upper bound of the eSTL formula
    """

    def __init__(
        self,
        low: float = 0.0,
        high: float = 0.0,
        t_low: int = 0,
        t_high: int = 0,
    ):  
        self.low = low
        self.high = high
        self.t_low = t_low
        self.t_high = t_high

    def negation(self):
        """
        Negation of eSTL formula.
        """
        self.low, self.high = -self.high, -self.low
        self.t_low, self.t_high = self.t_high, self.t_low
        return self
    
    @staticmethod
    def conjunction(formulas):
        """
        Conjunction of eSTL formulas at a specific time.

        Parameters:
            list_formulas : dict of {time: eSTL formula}
        """
        formula_low = min(formulas.values(), key=lambda f: f.low) # min(f.low for f in formulas.values())
        formula_high  = min(formulas.values(), key=lambda f: f.high) #min(f.high for f in formulas.values())
        formula = EllipsoidalSignalTemporalLogic(formula_low.low, formula_high.high, formula_low.t_low, formula_high.t_high)
        return formula

    @staticmethod
    def disjunction(formulas):
        """
        disjunction of eSTL formulas at a specific time.

        Parameters:
            list_formulas : dict of {time: eSTL formula}
        """
        formula_low = max(formulas.values(), key=lambda f: f.low) # min(f.low for f in formulas.values())
        formula_high  = max(formulas.values(), key=lambda f: f.high) #min(f.high for f in formulas.values())
        formula = EllipsoidalSignalTemporalLogic(formula_low.low, formula_high.high, formula_low.t_low, formula_high.t_high)
        return formula
    
    @staticmethod
    def globally(formulas, time_horizon):
        """
        Globally operator for eSTL.

        Parameters:
            formulas : dict of {time: eSTL formula}
            time_horizon : iterable of time steps to consider
        """
        new_formulas = {t: formulas[t] for t in time_horizon if t in formulas}
        global_formula = EllipsoidalSignalTemporalLogic.conjunction(new_formulas)
        return global_formula

    @staticmethod
    def eventually(formulas, time_horizon):
        """
        eventually operator for eSTL.

        Parameters:
            formulas : dict of {time: eSTL formula}
            time_horizon : iterable of time steps to consider
        """
        new_formulas = {t: formulas[t] for t in time_horizon if t in formulas}
        global_formula = EllipsoidalSignalTemporalLogic.disjunction(new_formulas)
        return global_formula

    @staticmethod
    def eventually_globally(formulas, eventually_horizon=None, globally_horizon=None):
        """
        Eventually-Globally (FG) operator for eSTL.

        Parameters:
            formulas : dict of {time: eSTL formula} (the base signal)
            eventually_horizon : iterable of offsets for the 'F' operator (None for unbounded)
            globally_horizon : iterable of offsets for the 'G' operator (None for unbounded)
        """
        intermediate_results = {}

        # Ensure we process times in chronological order
        all_times = sorted(formulas.keys())

        # 1. Compute G[globally_horizon] for every valid starting time t
        for t in all_times:
            if globally_horizon is None:
                # Unbounded: Evaluate from t to the end of the trace
                target_g_times = [st for st in all_times if st >= t]
            else:
                # Bounded: Shift the provided horizon by t
                target_g_times = [t + offset for offset in globally_horizon]

            # Keep only the time steps that actually exist in the trace.
            # This prevents the boundary failure at the end of the trace.
            valid_g_times = [st for st in target_g_times if st in formulas]

            if valid_g_times:
                intermediate_results[t] = EllipsoidalSignalTemporalLogic.globally(
                    formulas, valid_g_times
                )

        # 2. Compute F[eventually_horizon] over the intermediate "Globally" results
        if eventually_horizon is None:
            # Unbounded: Evaluate over all computed intermediate results
            target_f_times = all_times
        else:
            target_f_times = eventually_horizon

        # Ensure we only ask the 'eventually' function to evaluate times we successfully computed
        valid_f_times = [st for st in target_f_times if st in intermediate_results]

        final_formula = EllipsoidalSignalTemporalLogic.eventually(
            intermediate_results, valid_f_times
        )

        return final_formula


class SignalTemporalLogic:
    """
    This class implements Signal Temporal Logic (STL)

    Attributes:
        phi : robustness of STL formula
    """

    def __init__(
            self,
            phi: float = 0.0,
            t_phi: int = 0,
    ):
        self.phi = phi
        self.t_phi = t_phi

    def negation(self):
        """
        Negation of eSTL formula.
        """
        self.phi = copy.deepcopy(-self.phi)
        return self

    @staticmethod
    def conjunction(formulas):
        """
        Conjunction of eSTL formulas at a specific time.

        Parameters:
            list_formulas : dict of {time: eSTL formula}
        """
        formula_rho = min(formulas.values(), key=lambda f: f.phi)
        formula = SignalTemporalLogic(formula_rho.phi, formula_rho.t_phi)
        return formula

    @staticmethod
    def disjunction(formulas):
        """
        Disjunction of eSTL formulas at a specific time.

        Parameters:
            list_formulas : dict of {time: eSTL formula}
        """
        formula_rho = max(formulas.values(), key=lambda f: f.phi)
        formula = SignalTemporalLogic(formula_rho.phi, formula_rho.t_phi)
        return formula

    @staticmethod
    def globally(formulas, time_horizon):
        """
        Globally operator for eSTL.

        Parameters:
            formulas : dict of {time: eSTL formula}
            time_horizon : iterable of time steps to consider
        """
        new_formulas = {t: formulas[t] for t in time_horizon if t in formulas}
        global_formula = SignalTemporalLogic.conjunction(new_formulas)
        return global_formula


    @staticmethod
    def eventually(formulas, time_horizon):
        """
        eventually operator for eSTL.

        Parameters:
            formulas : dict of {time: eSTL formula}
            time_horizon : iterable of time steps to consider
        """
        new_formulas = {t: formulas[t] for t in time_horizon if t in formulas}
        global_formula = SignalTemporalLogic.disjunction(new_formulas)
        return global_formula

    @staticmethod
    def eventually_globally(formulas, eventually_horizon=None, globally_horizon=None):
        """
        Eventually-Globally (FG) operator for STL.

        Logic:
        1. For each t in the trace, calculate the 'Globally'
           satisfaction over the globally_horizon starting at t.
        2. Take the maximum (Eventually/Disjunction) of those results.
        """
        g_results_at_t = {}

        # Ensure we process times in chronological order
        all_times = sorted(formulas.keys())

        # 1. Compute G[globally_horizon] for every valid starting time t
        for t in all_times:
            if globally_horizon is None:
                # Unbounded: Evaluate from t to the end of the trace
                target_g_times = [st for st in all_times if st >= t]
            else:
                # Bounded: Shift the provided horizon by t
                target_g_times = [t + offset for offset in globally_horizon]

            # Keep only the time steps that actually exist in the trace
            # This safely handles the end-of-trace boundary condition
            valid_g_times = [st for st in target_g_times if st in formulas]

            if valid_g_times:
                g_results_at_t[t] = SignalTemporalLogic.globally(
                    formulas, valid_g_times
                )

        # 2. Compute F[eventually_horizon] over the intermediate "Globally" results
        if eventually_horizon is None:
            # Unbounded: Evaluate over all computed intermediate results
            target_f_times = all_times
        else:
            target_f_times = eventually_horizon

        # Ensure we only ask the 'eventually' function to evaluate times we successfully computed
        valid_f_times = [st for st in target_f_times if st in g_results_at_t]

        final_formula = SignalTemporalLogic.eventually(
            g_results_at_t, valid_f_times
        )

        return final_formula
       