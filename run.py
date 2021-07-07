from pyelastic import double_pendulum
import time

start = time.time()
pendulum = double_pendulum.ElasticPendulum(fps=24, t_end=25.0)
_ = pendulum.integrate()

for i in range(pendulum.x1.shape[0]):
    pendulum.save_frame(i, interpolate=False)

print(time.time() - start)

pendulum.make_movie()
