import os
import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.integrate import solve_ivp
from .pendulum import ElasticPendulum

ASSETS = os.path.abspath(".")


class Animation:
    """ """

    def __init__(
        self,
        alpha=None,
        beta=None,
        seed=None,
        fps=24,
        npends=1,
        tend=15,
        offset=0.05,
        filename="post.mp4",
    ):
        """ """
        self.dpi = 100
        self.size = 712
        self.fps = fps
        self.ns = 50
        self.s = 4
        self.tend = tend
        self.npends = npends
        self.filename = os.path.join(ASSETS, filename)

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
            self.beta = np.linspace(beta, beta + offset, npends)

    def _init_figure(self, size=712, dpi=100):
        """ """
        mini_x, maxi_x = np.inf, -np.inf
        mini_y, maxi_y = np.inf, -np.inf

        for pend in self.pendulums:
            if np.min([pend.x1, pend.x2]) < mini_x:
                mini_x = np.min([pend.x1, pend.x2])
            if np.max([pend.x1, pend.x2]) > maxi_x:
                maxi_x = np.max([pend.x1, pend.x2])
            if np.min([pend.y1, pend.y2]) < mini_y:
                mini_y = np.min([pend.y1, pend.y2])
            if np.max([pend.y1, pend.y2]) > maxi_y:
                maxi_y = np.max([pend.y1, pend.y2])

        if isinstance(size, tuple):
            fig = plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi)

        else:
            fig = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi)

        ax = plt.axes(xlim=[mini_x, maxi_x], ylim=[mini_y, maxi_y])
        ax.axis("off")

        if isinstance(size, tuple):
            fig.set_size_inches(size[0] / dpi, size[1] / dpi, forward=True)

        else:
            fig.set_size_inches(size / dpi, size / dpi, forward=True)

        fig.tight_layout()

        return fig, ax

    def init_single(self):
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

    def update_single(self, i):
        """ """
        self.line1.set_data([0, self.pendulum.x1[i]], [0, self.pendulum.y1[i]])
        self.dot1.set_data(self.pendulum.x1[i], self.pendulum.y1[i])
        self.line2.set_data(
            [self.pendulum.x1[i], self.pendulum.x2[i]],
            [self.pendulum.y1[i], self.pendulum.y2[i]],
        )
        self.dot2.set_data(self.pendulum.x2[i], self.pendulum.y2[i])
        self.dot3.set_data(0, 0)

        for j in range(self.ns):
            imin = i - (self.ns - j) * self.s
            if imin < 0:
                continue
            imax = imin + self.s + 1
            alpha = (j / self.ns) ** 2
            self.trace_lc1[j].set_data(
                self.pendulum.x1[imin:imax], self.pendulum.y1[imin:imax]
            )
            self.trace_lc1[j].set_alpha(alpha)
            self.trace_lc2[j].set_data(
                self.pendulum.x2[imin:imax], self.pendulum.y2[imin:imax]
            )
            self.trace_lc2[j].set_alpha(alpha)

        return self.line1, self.dot1, self.line2, self.dot2, self.dot3

    def animate_single(self, size=712, dpi=100, colors=("cyan", "magenta")):
        """ """
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
        (self.line1,) = ax.plot([], [], lw=2, color=colors[0], zorder=0)
        (self.dot1,) = ax.plot([], [], color=colors[0], marker="o", zorder=2)
        (self.line2,) = ax.plot([], [], lw=2, color=colors[1], zorder=0)
        (self.dot2,) = ax.plot([], [], color=colors[1], marker="o", zorder=2)
        (self.dot3,) = ax.plot([], [], color=colors[0], marker="o", zorder=2)
        self.trace_lc1 = []
        self.trace_lc2 = []
        for _ in range(self.ns):
            (trace1,) = ax.plot(
                [],
                [],
                c=colors[0],
                solid_capstyle="round",
                lw=1.5,
                alpha=0,
                zorder=0,
            )
            (trace2,) = ax.plot(
                [],
                [],
                c=colors[1],
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
            self.update_single,
            init_func=self.init_single,
            frames=self.pendulum.x1.shape[0],
            interval=0,
            blit=True,
            cache_frame_data=False,
        )

        anim.save(self.filename, fps=self.fps)

    def integrate(self, k1=None, k2=None, **pendulum_settings):
        """ """
        # Initialize Pendulums and Integrate
        self.pendulums = []

        if k1 is None:
            k1 = np.random.uniform(35, 55)

        if k2 is None:
            k2 = np.random.uniform(35, 55)

        for beta in self.beta:
            ds = ElasticPendulum(
                fps=self.fps,
                t_end=self.tend,
                alpha_0=self.alpha,
                beta_0=beta,
                k1=k1,
                k2=k2,
                **pendulum_settings
            )
            ds.integrate()
            self.pendulums.append(ds)

    def init_multiple(self):
        """ """
        self.anchor.set_data(0, 0)

        for pi in range(self.npends):
            self.linetop[pi].set_data([], [])
            self.dotmid[pi].set_data([], [])
            self.lines[pi].set_data([], [])
            self.dots[pi].set_data([], [])
            for j in range(self.ns):
                self.traces[pi][j].set_data([], [])

        return (self.linetop[0],)

    def update_multiple(self, i):
        """ """
        for pi in range(self.npends):
            self.linetop[pi].set_data(
                [0, self.pendulums[pi].x1[i]], [0, self.pendulums[pi].y1[i]]
            )
            self.dotmid[pi].set_data(self.pendulums[pi].x1[i], self.pendulums[pi].y1[i])
            self.dots[pi].set_data(self.pendulums[pi].x2[i], self.pendulums[pi].y2[i])
            self.lines[pi].set_data(
                [self.pendulums[pi].x1[i], self.pendulums[pi].x2[i]],
                [self.pendulums[pi].y1[i], self.pendulums[pi].y2[i]],
            )
            for j in range(self.ns):
                imin = i - (self.ns - j) * self.s
                if imin < 0:
                    continue
                imax = imin + self.s + 1
                alpha = (j / self.ns) ** 2
                self.traces[pi][j].set_data(
                    self.pendulums[pi].x2[imin:imax], self.pendulums[pi].y2[imin:imax]
                )
                self.traces[pi][j].set_alpha(alpha)

        return (self.linetop[0],)

    def animate_multiple(self, size=712, dpi=100, cmap=plt.cm.inferno):
        """ """
        self.integrate()
        fig, ax = self._init_figure()

        # Check if this is actually a colormap
        colors = cmap(np.linspace(0.25, 1, self.npends))

        # Plot Elements
        (self.anchor,) = ax.plot(
            [], [], color=(51 / 255, 59 / 255, 71 / 255, 1), marker="o", zorder=2
        )
        self.traces = []
        self.dots = []
        self.lines = []
        self.linetop = []
        self.dotmid = []
        for pi in range(self.npends):
            traces = []
            (linetop,) = ax.plot(
                [], [], lw=2, color=(51 / 255, 59 / 255, 71 / 255, 1), zorder=0
            )
            (dotmid,) = ax.plot([], [], color=colors[pi], marker="o", zorder=2)
            (dot,) = ax.plot([], [], color=colors[pi], marker="o", zorder=2)
            (line,) = ax.plot([], [], color=colors[pi], lw=2, zorder=2)
            self.dots.append(dot)
            self.lines.append(line)
            self.linetop.append(linetop)
            self.dotmid.append(dotmid)
            for _ in range(self.ns):
                (trace,) = ax.plot(
                    [],
                    [],
                    c=colors[pi],
                    solid_capstyle="round",
                    lw=2,
                    alpha=0,
                    zorder=0,
                )
                traces.append(trace)

            self.traces.append(traces)

        # Initialize Animation

        anim = animation.FuncAnimation(
            fig,
            self.update_multiple,
            init_func=self.init_multiple,
            frames=self.pendulums[0].x1.shape[0],
            interval=0,
            blit=True,
            cache_frame_data=False,
        )

        # Save animation
        anim.save(self.filename, fps=self.fps)

    def animate(
        self,
        size=712,
        dpi=100,
        cmap=plt.cm.inferno,
        colors=("cyan", "magenta"),
    ):
        """ """
        if self.npends > 1:
            self.animate_multiple(size=size, dpi=dpi, cmap=cmap)

        else:
            self.animate_single(size=size, dpi=dpi, colors=colors)
