# Springy Double Pendulum

Twitter bot that generates and posts videos of a double elastic pendulum. The bot
works by integrating the Lagrangian equations of motion for a two-dimensional double
spring pendulum using `scipy.integrate`. Videos can be quickly generated and simulations
flexibly allow a range of parameters for the pendulum and customizible pendulum and background colors.

![Example Simulation](assets/sim.gif)

# Installation

The install the bot simply clone this repository and run the following commands

```
git clone https://github.com/tyler-a-cox/elastic-pendulum
cd elastic-pendulum
python setup.py install
```

# Basic Usage

This package can either be run to post videos to twitter given the proper credentials or you can simply 
simulate a pendulum and save the video like so

```
from pyelastic.animate import Animation

anim = Animation(fps=60, tend=10)
anim.animate()
```
For more information on the physics involved in the simulation, check out the [blog post](https://tyleracox.xyz/blog/double-pendulum/) or you can follow the bot on Twitter at [@springy_bot](https://twitter/com/springy_bot).
