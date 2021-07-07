from pyelastic import double_pendulum
import time

start = time.time()
pendulum = double_pendulum.ElasticPendulum(fps=60, t_end=10.0)
_ = pendulum.integrate()

for i in range(pendulum.x1.shape[0] + 60):
    pendulum.save_frame(i)

print(time.time() - start)

pendulum.make_movie()
