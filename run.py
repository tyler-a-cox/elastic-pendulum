from pyelastic import double_pendulum
import time

start = time.time()
pendulum = double_pendulum.ElasticPendulum(fps=24, t_end=20)
_ = pendulum.integrate()

# for i in range(pendulum.x1.shape[0]):
#    pendulum.save_frame(i, interpolate=False)
pendulum.main_animate()
print(time.time() - start)

# pendulum.make_movie()
