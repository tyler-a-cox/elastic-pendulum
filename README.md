# Springy Double Pendulum Twitter Bot

<p align="center">
  <img src="assets/sim.gif" alt="animated" width="95%" height="60%"/>
</p>

Elastic Pendulum is a bot that generates and posts videos of double elastic pendulums to Twitter twice a day. A double elastic pendulum (or double spring pendulum) is a variant of a common system in mechanics known as the double pendulum, which is essentially a pendulum attached to the end of another pendulum. These types of systems are particularly interesting because their motion is complex and cannot be solved for using standard Newtonian mechanics. The main difference between a double pendulum and a double spring pendulum is that the fixed rods commonly seen in double pendulum problems are replaced with springs. This minor change leads to even more chaotic behavior than the standard double pendulums. The gif above shows 5 double pendulums simulated with slightly different starting angles leading to different simulated trajectories. 

The bot works by integrating the Lagrangian equations of motion for a two-dimensional double spring pendulum using `scipy.integrate`. Videos can be quickly generated and simulations flexibly allow a range of parameters for the double pendulum system and customizible pendulum and background colors.

# Installation

To install the bot simply clone this repository and run the following commands in the terminal

```
git clone https://github.com/tyler-a-cox/elastic-pendulum
cd elastic-pendulum
python setup.py install
```
or you can install directly from the git repo

`pip install git+https://github.com/tyler-a-cox/elastic-pendulum`

# Basic Usage

This package can either be run to post videos to twitter given the proper credentials or you can simply 
simulate a pendulum and save the video like so

```
from pyelastic.animate import Animation

anim = Animation(fps=60, tend=10, filename='example.mp4')
anim.animate()
```
For more information on the physics involved in the simulation, check out the [blog post](https://tyleracox.xyz/blog/double-pendulum/) or you can follow the bot on Twitter at [@springy_bot](https://twitter.com/springy_bot).
