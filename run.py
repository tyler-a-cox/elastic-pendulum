from pyelastic.animate import Animation
import time
import pylab as plt

timings = {}

start = time.time()
anim = Animation(fps=30, tend=10, npends=5, offset=0.0001)
anim.main_n_animate(cmap=plt.cm.inferno)


print("Animation time", time.time() - start)
print("FPS:", 300 / (time.time() - start))

# pendulum.make_movie()
