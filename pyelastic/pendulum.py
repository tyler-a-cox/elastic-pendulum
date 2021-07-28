import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from .settings import *


class ElasticPendulum:
    """Animate

    Args:
        save_movie : boolean, default=True

    Returns:
        None
    """

    def __init__(self, **kwargs):
        """Animate

        Args:
            alpha_0 : boolean, default=True
            beta_0 : boolean, default=True
            alpha_1 : boolean, default=True
            beta_1 : boolean, default=True
            k1 : boolean, default=True
            k2 : boolean, default=True
            l1 : boolean, default=True
            l2 : boolean, default=True
            m1 : boolean, default=True
            m2 : boolean, default=True
            a0 : boolean, default=True
            b0 : boolean, default=True
            a1 : boolean, default=True
            b1 : boolean, default=True
            t_end : boolean, default=True
            fps : boolean, default=True
        """
        prop_defaults = {
            "alpha_0": np.random.uniform(-np.pi, np.pi),
            "beta_0": np.random.uniform(-np.pi, np.pi),
            "alpha_1": 0.0,
            "beta_1": 0.0,
            "k1": np.random.uniform(35, 55),
            "k2": np.random.uniform(35, 55),
            "l1": 1.0,
            "l2": 1.0,
            "m1": 1.0,
            "m2": 1.0,
            "a0": 1.0,
            "b0": 1.0,
            "a1": 1.0,
            "b1": 1.0,
            "t_end": 2,
            "fps": 24,
            "g": GRAVITY,
        }

        for (prop, default) in prop_defaults.iteritems():
            setattr(self, prop, kwargs.get(prop, default))

        self.dt = 1.0 / self.fps
        self.t_eval = np.arange(0, self.t_end, self.dt)

    def _spherical_to_cartesian(self, array, interpolate=True):
        """Transforms from 2D spherical coordinate system to a cartesian coordinate system

        Args:
            array : np.ndarray
                Output array from integration function in spherical coordinates

            interpolate : boolean, default=True


        Returns:
            None
        """
        x1 = array[:, 2] * np.sin(array[:, 0])
        x2 = x1 + array[:, 3] * np.sin(array[:, 1])
        y1 = -array[:, 2] * np.cos(array[:, 0])
        y2 = y1 - array[:, 3] * np.cos(array[:, 1])

        if interpolate:
            self.fx1 = interp1d(np.arange(0, x1.shape[0]), x1)
            self.fy1 = interp1d(np.arange(0, x1.shape[0]), y1)
            self.fx2 = interp1d(np.arange(0, x1.shape[0]), x2)
            self.fy2 = interp1d(np.arange(0, x1.shape[0]), y2)

        return x1, x2, y1, y2

    def _alpha_pp(self, t, Y):
        """ """
        alpha_0, alpha_1, beta_0, beta_1, a0, a1, b0, _ = Y
        return -(
            self.g * self.m1 * np.sin(alpha_0)
            - self.k2 * self.l2 * np.sin(alpha_0 - beta_0)
            + self.k2 * b0 * np.sin(alpha_0 - beta_0)
            + 2 * self.m1 * a1 * alpha_1
        ) / (self.m1 * a0)

    def _beta_pp(self, t, Y):
        """ """
        alpha_0, alpha_1, beta_0, beta_1, a0, a1, b0, b1 = Y
        return (
            -self.k1 * self.l1 * np.sin(alpha_0 - beta_0)
            + self.k1 * a0 * np.sin(alpha_0 - beta_0)
            - 2.0 * self.m1 * b1 * beta_1
        ) / (self.m1 * b0)

    def _a_pp(self, t, Y):
        """ """
        alpha_0, alpha_1, beta_0, beta_1, a0, a1, b0, b1 = Y
        return (
            self.k1 * self.l1
            + self.g * self.m1 * np.cos(alpha_0)
            - self.k2 * self.l2 * np.cos(alpha_0 - beta_0)
            + self.k2 * b0 * np.cos(alpha_0 - beta_0)
            + a0 * (-self.k1 + self.m1 * alpha_1 ** 2)
        ) / self.m1

    def _b_pp(self, t, Y):
        """ """
        alpha_0, alpha_1, beta_0, beta_1, a0, a1, b0, b1 = Y
        return (
            self.k2 * self.l2 * self.m1
            + self.k2 * self.l2 * self.m2 * np.cos(alpha_0 - beta_0)
            + self.k1 * self.m2 * a0 * np.cos(alpha_0 - beta_0)
            - b0 * (self.k2 * (self.m1 + self.m2) - self.m1 * self.m2 * beta_1 ** 2)
        ) / (self.m1 * self.m2)

    def _lagrangian(self, t, Y):
        """Set of differential equations to integrate to solve the equations of motion
        for the pendulum masses. Incorporates

        Args:
            t : np.ndarray
                Evaluation time array
            Y : np.ndarray
                Initial conditions of the pendulum masses
        Returns:
            list :
                Evaluation of the differential equations
        """
        return [
            Y[1],
            self._alpha_pp(t, Y),
            Y[3],
            self._beta_pp(t, Y),
            Y[5],
            self._a_pp(t, Y),
            Y[7],
            self._b_pp(t, Y),
        ]

    def integrate(self, method="LSODA", interpolate=True):
        """Animate

        Args:
            save_movie : boolean, default=True

        Returns:
            None
        """
        Y0 = [
            self.alpha_0,
            self.alpha_1,
            self.beta_0,
            self.beta_1,
            self.a0,
            self.a1,
            self.b0,
            self.b1,
        ]
        self.solution = solve_ivp(
            self._lagrangian, [0, self.t_end], Y0, t_eval=self.t_eval, method=method
        )
        self.x1, self.x2, self.y1, self.y2 = self._spherical_to_cartesian(
            self.solution.y[[0, 2, 4, 6]].T, interpolate=interpolate
        )
