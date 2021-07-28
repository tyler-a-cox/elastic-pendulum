import os
import pylab as plt
import cmasher as cm


# Plot Settings
plt.rc("text", usetex=False)
plt.style.use("dark_background")
COLORMAPS = [
    cm.neon,
    cm.bubblegum,
    cm.voltage,
    cm.horizon,
    cm.fall,
    cm.sunburst,
    cm.flamingo,
    cm.eclipse,
    cm.lilac,
    cm.lavender,
    cm.sepia,
    plt.cm.jet,
]

# Twitter Settings
hasttags = ["#science", "#scicomm", "#physics", "#sciart"]
tags = " ".join(hasttags)

# Physics Settings
GRAVITY = 9.81

# Save Settings
ASSETS = os.path.abspath(".")
