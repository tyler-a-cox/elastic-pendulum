import os
import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from multiprocessing import cpu_count, Pool

from ._keys import FIG_DIR, VID_DIR

plt.rc("text", usetex=False)
plt.style.use("dark_background")


class ElasticPendulum:
    """Animate

    Args:
        save_movie : boolean, default=True

    Returns:
        None
    """

    def __init__(self, alpha_0=None, beta_0=None, t_end=2, fps=24, cores=None):
        """Animate

        Args:
            save_movie : boolean, default=True

        Returns:
            None
        """
        self.g = 9.81
        if alpha_0 is not None:
            self.alpha_0 = alpha_0
        else:
            self.alpha_0 = np.random.uniform(-np.pi, np.pi)

        if beta_0 is not None:
            self.beta_0 = beta_0
        else:
            self.beta_0 = np.random.uniform(-np.pi, np.pi)

        self.fig_dir = FIG_DIR
        self.vid_dir = VID_DIR

        if not os.path.exists(self.fig_dir):
            os.mkdir(self.fig_dir)

        self.alpha_1 = 0.0
        self.beta_1 = 0.0

        self.a0 = 1.0
        self.b0 = 1.0
        self.a1 = 0.0
        self.b1 = 0.0
        self.ns = 50
        self.s = 4
        self.t_end = t_end

        #
        self.l1 = 1.0
        self.l2 = 1.0
        self.m1 = 1.0
        self.m2 = 1.0
        self.k1 = np.random.uniform(35, 55)
        self.k2 = np.random.uniform(35, 55)
        self.fps = fps
        self.dt = 1.0 / self.fps

        self.t_eval = np.arange(0, self.t_end, self.dt)
        if cores is None:
            self.cores = cpu_count()
        else:
            self.cores = cores

    def _alpha_pp(self, t, Y):
        """Animate

        Args:
            save_movie : boolean, default=True

        Returns:
            None
        """
        # Y[0] = alpha_1, Y[1] = alpha_0
        alpha_0, alpha_1, beta_0, beta_1, a0, a1, b0, _ = Y
        return -(
            self.g * self.m1 * np.sin(alpha_0)
            - self.k2 * self.l2 * np.sin(alpha_0 - beta_0)
            + self.k2 * b0 * np.sin(alpha_0 - beta_0)
            + 2 * self.m1 * a1 * alpha_1
        ) / (self.m1 * a0)

    def _beta_pp(self, t, Y):
        """Animate

        Args:
            save_movie : boolean, default=True

        Returns:
            None
        """
        # Y[0] = beta_1, Y[1] = beta_0
        alpha_0, alpha_1, beta_0, beta_1, a0, a1, b0, b1 = Y
        return (
            -self.k1 * self.l1 * np.sin(alpha_0 - beta_0)
            + self.k1 * a0 * np.sin(alpha_0 - beta_0)
            - 2.0 * self.m1 * b1 * beta_1
        ) / (self.m1 * b0)

    def _a_pp(self, t, Y):
        """Animate

        Args:
            save_movie : boolean, default=True

        Returns:
            None
        """
        # Y[0] = a1, Y[1] = a0
        alpha_0, alpha_1, beta_0, beta_1, a0, a1, b0, b1 = Y
        return (
            self.k1 * self.l1
            + self.g * self.m1 * np.cos(alpha_0)
            - self.k2 * self.l2 * np.cos(alpha_0 - beta_0)
            + self.k2 * b0 * np.cos(alpha_0 - beta_0)
            + a0 * (-self.k1 + self.m1 * alpha_1 ** 2)
        ) / self.m1

    def _b_pp(self, t, Y):
        """Animate

        Args:
            save_movie : boolean, default=True

        Returns:
            None
        """
        # Y[0] = b1, Y[1] = b0
        alpha_0, alpha_1, beta_0, beta_1, a0, a1, b0, b1 = Y
        return (
            self.k2 * self.l2 * self.m1
            + self.k2 * self.l2 * self.m2 * np.cos(alpha_0 - beta_0)
            + self.k1 * self.m2 * a0 * np.cos(alpha_0 - beta_0)
            - b0 * (self.k2 * (self.m1 + self.m2) - self.m1 * self.m2 * beta_1 ** 2)
        ) / (self.m1 * self.m2)

    def _inte(self, t, Y):
        """Animate

        Args:
            save_movie : boolean, default=True

        Returns:
            None
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

    def integrate(self, method="LSODA"):
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
            self._inte, [0, self.t_end], Y0, t_eval=self.t_eval, method=method
        )
        return self.cartesian(self.solution.y[[0, 2, 4, 6]].T)

    def cartesian(self, array):
        """Animate

        Args:
            save_movie : boolean, default=True

        Returns:
            None
        """
        self.x1 = array[:, 2] * np.sin(array[:, 0])
        self.x2 = self.x1 + array[:, 3] * np.sin(array[:, 1])
        self.y1 = -array[:, 2] * np.cos(array[:, 0])
        self.y2 = self.y1 - array[:, 3] * np.cos(array[:, 1])

        self.fx1 = interp1d(np.arange(0, self.x1.shape[0]), self.x1)
        self.fy1 = interp1d(np.arange(0, self.x1.shape[0]), self.y1)
        self.fx2 = interp1d(np.arange(0, self.x1.shape[0]), self.x2)
        self.fy2 = interp1d(np.arange(0, self.x1.shape[0]), self.y2)
        return self.x1, self.y1, self.x2, self.y2

    def single_init(self):
        """ """
        self.line1.set_data([], [])
        self.dot1.set_data([], [])
        self.line2.set_data([], [])
        self.dot2.set_data([], [])
        self.dot3.set_data([], [])
        for j in range(self.ns):
            self.trace_lc1[j].set_data([], [])
            self.trace_lc2[j].set_data([], [])
        return self.line1, self.dot1, self.line2, self.dot2, self.dot3

    def single_animate(self, i):
        """ """
        self.line1.set_data([0, self.x1[i]], [0, self.y1[i]])
        self.dot1.set_data(self.x1[i], self.y1[i])
        self.line2.set_data(
            [self.x1[i], self.x2[i]],
            [self.y1[i], self.y2[i]],
        )
        self.dot2.set_data(self.x2[i], self.y2[i])
        self.dot3.set_data(0, 0)

        for j in range(self.ns):
            imin = i - (self.ns - j) * self.s
            if imin < 0:
                continue
            imax = imin + self.s + 1
            alpha = (j / self.ns) ** 2
            self.trace_lc1[j].set_data(self.x1[imin:imax], self.y1[imin:imax])
            self.trace_lc1[j].set_alpha(alpha)
            self.trace_lc2[j].set_data(self.x2[imin:imax], self.y2[imin:imax])
            self.trace_lc2[j].set_alpha(alpha)

        return self.line1, self.dot1, self.line2, self.dot2, self.dot3

    def main_animate(self, size=800, dpi=100):
        """ """
        self.fig = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi)
        m1 = np.max([self.x1, self.x2])
        m2 = np.max([self.y1, self.y2])
        if m1 < 0:
            m1 = 2
        if m2 < 0:
            m2 = 2

        ax = plt.axes(
            xlim=[np.min([self.x1, self.x2]), m1], ylim=[np.min([self.y1, self.y2]), m2]
        )
        ax.axis("off")
        (self.line1,) = ax.plot([], [], lw=2, color="cyan", zorder=0)
        (self.dot1,) = ax.plot([], [], color="cyan", marker="o", zorder=2)
        (self.line2,) = ax.plot([], [], lw=2, color="magenta", zorder=0)
        (self.dot2,) = ax.plot([], [], color="magenta", marker="o", zorder=2)
        (self.dot3,) = ax.plot([], [], color="cyan", marker="o", zorder=2)
        self.trace_lc1 = []
        self.trace_lc2 = []
        for _ in range(self.ns):
            (trace1,) = ax.plot(
                [],
                [],
                c="cyan",
                solid_capstyle="round",
                lw=1.5,
                alpha=0,
                zorder=0,
            )
            (trace2,) = ax.plot(
                [],
                [],
                c="magenta",
                solid_capstyle="round",
                lw=1.5,
                alpha=0,
                zorder=0,
            )
            self.trace_lc1.append(trace1)
            self.trace_lc2.append(trace2)

        self.fig.set_size_inches(size / dpi, size / dpi, forward=True)
        self.fig.tight_layout()

        anim = animation.FuncAnimation(
            self.fig,
            self.single_animate,
            init_func=self.single_init,
            frames=self.x1.shape[0],
            interval=0,
            blit=True,
            cache_frame_data=False,
        )

        # anim.save(
        #    "basic_animation.mp4", fps=self.fps, extra_args=["-vcodec", "libx264"]
        # )
        anim.save("example_sim.gif", fps=self.fps, writer="Pillow")

    def n_init(self):
        """ """
        self.line1.set_data([], [])
        self.dot1.set_data([], [])
        self.dot3.set_data([], [])

        for j in range(self.ns):
            self.trace_lc1[j].set_data([], [])
            self.trace_lc2[j].set_data([], [])

        return (self.line1,)

    def n_animate(self, i):
        """ """
        self.line1.set_data([0, self.x1[i]], [0, self.y1[i]])
        self.dot1.set_data(self.x1[i], self.y1[i])
        self.dot3.set_data(0, 0)

        for j in range(self.ns):
            imin = i - (self.ns - j) * self.s
            if imin < 0:
                continue
            imax = imin + self.s + 1
            alpha = (j / self.ns) ** 2
            self.trace_lc1[j].set_data(self.x1[imin:imax], self.y1[imin:imax])
            self.trace_lc1[j].set_alpha(alpha)
            self.trace_lc2[j].set_data(self.x2[imin:imax], self.y2[imin:imax])
            self.trace_lc2[j].set_alpha(alpha)

        return (self.line1,)

    def main_n_animate(self, size=800, dpi=100):
        """ """
        self.fig = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi)
        m1 = np.max([self.x1, self.x2])
        m2 = np.max([self.y1, self.y2])
        if m1 < 0:
            m1 = 2
        if m2 < 0:
            m2 = 2

        ax = plt.axes(
            xlim=[np.min([self.x1, self.x2]), m1], ylim=[np.min([self.y1, self.y2]), m2]
        )
        ax.axis("off")
        (self.line1,) = ax.plot([], [], lw=2, color="white", zorder=0)
        (self.dot1,) = ax.plot([], [], color="white", marker="o", zorder=2)
        (self.line2,) = ax.plot([], [], lw=2, color="magenta", zorder=0)
        (self.dot2,) = ax.plot([], [], color="magenta", marker="o", zorder=2)
        (self.dot3,) = ax.plot([], [], color="white", marker="o", zorder=2)
        self.trace_lc1 = []
        self.trace_lc2 = []
        for _ in range(self.ns):
            (trace1,) = ax.plot(
                [],
                [],
                c="cyan",
                solid_capstyle="round",
                lw=1.5,
                alpha=0,
                zorder=0,
            )
            (trace2,) = ax.plot(
                [],
                [],
                c="magenta",
                solid_capstyle="round",
                lw=1.5,
                alpha=0,
                zorder=0,
            )
            self.trace_lc1.append(trace1)
            self.trace_lc2.append(trace2)

        self.fig.set_size_inches(size / dpi, size / dpi, forward=True)
        self.fig.tight_layout()

        anim = animation.FuncAnimation(
            self.fig,
            self.single_animate,
            init_func=self.single_init,
            frames=self.x1.shape[0],
            interval=0,
            blit=True,
            cache_frame_data=False,
        )

        # anim.save(
        #    "basic_animation.mp4", fps=self.fps, extra_args=["-vcodec", "libx264"]
        # )
        anim.save("example_sim.gif", fps=self.fps, writer="pillow")
