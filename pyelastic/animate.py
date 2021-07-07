import os
import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from multiprocessing import cpu_count, Pool
from .double_pendulum import ElasticPendulum

VID_DIR = "/Users/tyler/Projects/elastic-pendulum/assets/"
FIG_DIR = "/Users/tyler/Projects/elastic-pendulum/assets/"


class Animation:
    """Animate

    Args:
        save_movie : boolean, default=True

    Returns:
        None
    """

    def __init__(
        self, alpha=None, beta=None, seed=None, cores=None, fps=30, pendulums=1
    ):
        self.dpi = 100
        self.size = 712
        self.fig_dir = FIG_DIR
        self.vid_dir = VID_DIR
        self.fps = fps
        if seed is not None:
            np.random.seed(seed)

        if alpha is not None:
            self.alpha = alpha
        else:
            self.alpha = np.random.uniform(-np.pi, np.pi)

        if beta is not None:
            self.beta = beta
        else:
            beta = np.random.uniform(-np.pi, np.pi)
            self.beta = np.linspace(beta, beta + 0.05, pendulums)

        if cores is not None:
            self.cores = cores
        else:
            self.cores = cpu_count()

    def _plot_settings(self, x):
        """Animate

        Args:
            save_movie : boolean, default=True

        Returns:
            None
        """
        colors = np.zeros((x.shape[0], 4))
        colors[:, 2] = 1
        alpha = np.linspace(0.2, 0.8, x.shape[0]) ** 2.0
        colors[:, 3] = alpha
        return colors

    def plot_frame(self, d, i, trace=True, axes_off=True):
        """Animate

        Args:
            save_movie : boolean, default=True

        Returns:
            None
        """
        colors = self._plot_settings(d.x1[:i])

        if trace:
            plt.scatter(d.x2[:i], d.y2[:i], color=colors[:i], s=2.0, zorder=0)

        if axes_off:
            plt.axis("off")

        plt.plot(
            [0, d.x1[i]],
            [0, d.y1[i]],
            color="black",
            zorder=1,
            linewidth=1.0,
            alpha=0.7,
        )
        plt.plot(
            [d.x1[i], d.x2[i]],
            [d.y1[i], d.y2[i]],
            color="blue",
            zorder=1,
            linewidth=1.0,
            alpha=0.7,
        )
        plt.scatter(
            [0, d.x1[i], d.x2[i]],
            [0, d.y1[i], d.y2[i]],
            color=((0, 0, 0, 1), (0, 0, 0, 1), (0, 0, 1, 1)),
            zorder=2,
        )

    def clear_figs(self):
        """Animate

        Args:
            save_movie : boolean, default=True

        Returns:
            None
        """
        figs = glob.glob(os.path.join(self.fig_dir, "*png"))
        for f in figs:
            os.remove(f)

    def main(self, i):
        """Animate

        Args:
            save_movie : boolean, default=True

        Returns:
            None
        """
        fig = plt.figure(
            figsize=(self.size / self.dpi, self.size / self.dpi), dpi=self.dpi
        )
        for pendulum in self.pendulums:
            self.plot_frame(pendulum, i)

        fig.set_size_inches(self.size / self.dpi, self.size / self.dpi, forward=True)

        plt.xlim([self.mini_x * 1.05, self.maxi_x * 1.05])
        plt.ylim([self.mini_y * 1.05, self.maxi_y * 1.05])

        fig.tight_layout()
        plt.savefig(os.path.join(self.fig_dir, str(i).zfill(5) + ".png"), dpi=self.dpi)
        plt.clf()
        plt.close()

    def save_movie(self):
        """Animate

        Args:
            save_movie : boolean, default=True

        Returns:
            None
        """
        dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.fname = os.path.join(self.vid_dir, "pend_{}.mp4".format(dt))
        figs = os.path.join(self.fig_dir, "%05d.png")
        os.system(
            "ffmpeg -r {} -f image2 -s 1920x1080 -i {} -vcodec \
                    libx264 -crf 25  -pix_fmt yuv420p {}".format(
                self.fps, figs, self.fname
            )
        )

    def animate(self, save_movie=True):
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

        self.clear_figs()
        pool = Pool(processes=self.cores)
        _ = pool.map(self.main, np.arange(self.pendulums[0].x1.shape[0]))

        if save_movie:
            self.save_movie()
