from pyelastic import double_pendulum
import time

timings = {}

start = time.time()
pendulum = double_pendulum.ElasticPendulum(fps=24, t_end=10)
_ = pendulum.integrate()
endint = time.time()
timings["integration"] = endint - start

# for i in range(pendulum.x1.shape[0]):
#    pendulum.save_frame(i, interpolate=False)
pendulum.main_animate()
anttime = time.time()
timings["animation"] = anttime - endint
print("Integration time", timings["integration"])
print("Animation time", timings["animation"])
print("FPS:", 240 / timings["animation"])

# pendulum.make_movie()
