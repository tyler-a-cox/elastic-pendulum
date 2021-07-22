import time
import cmasher as cm
import pylab as plt
from pyelastic.animate import Animation

cmaps = [cm.neon, cm.bubblegum]

timings = {}

start = time.time()
# Multi
anim = Animation(fps=24, tend=8, npends=5, offset=0.00001)
anim.main_n_animate(cmap=cm.bubblegum)

# Single
# anim = Animation(fps=30, tend=10)
# anim.main_animate()


print("Animation time", time.time() - start)
print("FPS:", 300 / (time.time() - start))

# pendulum.make_movie()
