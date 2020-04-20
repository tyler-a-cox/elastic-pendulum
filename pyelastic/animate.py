import os
import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from multiprocessing import cpu_count, Pool
from double_pendulum import DoubleSpring

def _plot_settings(x):
        colors = np.zeros((x.shape[0], 4))
        colors[:, 2] = 1
        alpha = np.linspace(0.2, 0.8, x.shape[0]) ** 2.
        colors[:, 3] = alpha
        return colors


def save_frame(fig, d, i, trace=True, axes_off=True):
    colors = _plot_settings(d.x1[:i])

    if trace:
        plt.scatter(d.x2[:i], d.y2[:i], color=colors[:i], s=2.,
                    zorder=0)

    if axes_off:
        plt.axis('off')

    plt.plot([0, d.x1[i]], [0, d.y1[i]], color='black',
             zorder=1, linewidth=1., alpha=0.7)
    plt.plot([d.x1[i], d.x2[i]], [d.y1[i], d.y2[i]],
             color='blue', zorder=1, linewidth=1., alpha=0.7)
    plt.scatter([0, d.x1[i], d.x2[i]], [0, d.y1[i], d.y2[i]],
                color=((0,0,0,1),(0,0,0,1),(0, 0, 1, 1)), zorder=2)


def clear_figs(fig_cache):
    """
    """
    figs = glob.glob(os.path.join(fig_cache, '*png'))
    for f in figs:
        os.remove(f)


def main(i):
    """
    """
    dpi = 100
    size = 712
    fig_cache = '_figs'
    fig = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi)
    for d in ds_s:
        save_frame(fig, d, i)
    fig.set_size_inches(size/dpi, size/dpi, forward=True)
    plt.xlim([mini_x * 1.05, maxi_x * 1.05])
    plt.ylim([mini_y * 1.05, maxi_y * 1.05])
    fig.tight_layout()
    plt.savefig(os.path.join(fig_cache, str(i).zfill(5) + '.png'), dpi=dpi)
    plt.clf()
    plt.close()

if __name__ == '__main__':
    ps = np.random.uniform(-np.pi, np.pi, size=2)
    alpha = ps[0]
    beta = np.linspace(ps[1], ps[1] + 0.15, 10)
    ds_s = []
    fps = 60

    for b in beta:
        ds = DoubleSpring(dt=1. / fps, t_end=15., alpha_0=alpha, beta_0=b)
        _ = ds.integrate()
        ds_s.append(ds)

    mini_x, maxi_x = 0, 1
    mini_y, maxi_y = 0, 1

    for d in ds_s:
        if np.min([d.x1, d.x2]) < mini_x:
            mini_x = np.min([d.x1, d.x2])
        if np.max([d.x1, d.x2]) > maxi_x:
            maxi_x = np.max([d.x1, d.x2])
        if np.min([d.y1, d.y2]) < mini_y:
            mini_y = np.min([d.y1, d.y2])
        if np.max([d.y1, d.y2]) > maxi_y:
            maxi_y = np.max([d.y1, d.y2])

    fig_cache = '_figs'
    vid_cache = '_videos'
    clear_figs(fig_cache)
    pool = Pool(processes=cpu_count())
    _ = pool.map(main, np.arange(ds_s[0].x1.shape[0]))
    t = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    fname = os.path.join(vid_cache, 'pend_{}.mp4'.format(t))
    figs = os.path.join(fig_cache, '%05d.png')
    os.system('ffmpeg -r {} -f image2 -s 1920x1080 -i {} -vcodec \
                libx264 -crf 25  -pix_fmt yuv420p {}'.format(fps, figs, fname))
