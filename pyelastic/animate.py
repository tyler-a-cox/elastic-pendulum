import os
import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from multiprocessing import cpu_count, Pool
from .double_pendulum import ElasticPendulum

VID_DIR = "assets"
plt.rc("text", usetex=False)
plt.style.use("dark_background")


class Animation:
    """Animate

    Args:
        save_movie : boolean, default=True

    Returns:
        None
    """

    def __init__(
        self, alpha=None, beta=None, seed=None, cores=None, fps=24, np=1, tend=15
    ):
        self.dpi = 100
        self.size = 712
        self.fig_dir = FIG_DIR
        self.vid_dir = VID_DIR
        self.fps = fps
        self.ns = 50
        self.s = 4

        if seed:
            np.random.seed(seed)

        if alpha:
            self.alpha = alpha
        else:
            self.alpha = np.random.uniform(-np.pi, np.pi)

        if beta:
            self.beta = beta
        else:
            beta = np.random.uniform(-np.pi, np.pi)
            self.beta = np.linspace(beta, beta + 0.05, pendulums)

        if cores:
            self.cores = cores
        else:
            self.cores = cpu_count()

    def animate_npends(self, save_movie=True):
        """Animate

        Args:
            save_movie : boolean, default=True

        Returns:
            None
        """
        self.pendulums = []

        for b in self.beta:
            ds = ElasticPendulum(
                dt=1.0 / self.fps, t_end=15.0, alpha_0=self.alpha, beta_0=b
            )
            _ = ds.integrate()
            self.pendulums.append(ds)

        self.mini_x, self.maxi_x = 0, 1
        self.mini_y, self.maxi_y = 0, 1

        for d in self.pendulums:
            if np.min([d.x1, d.x2]) < self.mini_x:
                self.mini_x = np.min([d.x1, d.x2])
            if np.max([d.x1, d.x2]) > self.maxi_x:
                self.maxi_x = np.max([d.x1, d.x2])
            if np.min([d.y1, d.y2]) < self.mini_y:
                self.mini_y = np.min([d.y1, d.y2])
            if np.max([d.y1, d.y2]) > self.maxi_y:
                self.maxi_y = np.max([d.y1, d.y2])

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
        self.line1.set_data([0, self.pendulum.x1[i]], [0, self.pendulum.y1[i]])
        self.dot1.set_data(self.pendulum.x1[i], self.pendulum.y1[i])
        self.line2.set_data(
            [self.pendulum.x1[i], self.pendulum.x2[i]],
            [self.pendulum.y1[i], self.pendulum.y2[i]],
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

    def main_animate(self, size=800, dpi=100, format="gif"):
        """ """
        assert format in ["gif", "mp4"], "Not a supported format"
        self.pendulum = ElasticPendulum(fps=self.fps, t_end=self.tend)
        _ = self.pendulum.integrate()

        self.fig = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi)
        m1 = np.max([self.pendulum.x1, self.pendulum.x2])
        m2 = np.max([self.pendulum.y1, self.pendulum.y2])
        if m1 < 0:
            m1 = 2
        if m2 < 0:
            m2 = 2

        ax = plt.axes(
            xlim=[np.min([self.pendulum.x1, self.pendulum.x2]), m1],
            ylim=[np.min([self.pendulum.y1, self.pendulum.y2]), m2],
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
            frames=pendulum.x1.shape[0],
            interval=0,
            blit=True,
            cache_frame_data=False,
        )

        if format == "gif":
            anim.save(os.path.join(VID_DIR, "sim.gif"), fps=self.fps)

        else:
            anim.save(
                os.path.join(VID_DIR, "sim.mp4"),
                fps=self.fps,
                extra_args=["-vcodec", "libx264"],
            )

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

    def main_n_animate(self, size=800, dpi=100, format="gif"):
        """ """
        assert format in ["gif", "mp4"], "Not a supported format"
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

        if format == "gif":
            anim.save(os.path.join(VID_DIR, "sim.gif"), fps=self.fps)

        else:
            anim.save(
                os.path.join(VID_DIR, "sim.mp4"),
                fps=self.fps,
                extra_args=["-vcodec", "libx264"],
            )
