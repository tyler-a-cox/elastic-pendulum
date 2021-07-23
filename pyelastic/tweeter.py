import os
import sys
import numpy as np
import cmasher as cm
import pylab as plt
from twython import Twython
from .animate import Animation

try:
    from ._keys import API_KEY, API_SECRET_KEY, ACCESS_TOKEN, ACCESS_SECRET_TOKEN

except ImportError:
    API_KEY = sys.getenv("API_KEY")
    API_SECRET_KEY = sys.getenv("API_SECRET_KEY")
    ACCESS_TOKEN = sys.getenv("ACCESS_TOKEN")
    ACCESS_SECRET_TOKEN = sys.getenv("ACCESS_SECRET_TOKEN")

cmaps = [
    cm.neon,
    cm.bubblegum,
    cm.voltage,
    cm.horizon,
    cm.fall,
    cm.sunburst,
    cm.flamingo,
    cm.eclipse,
    plt.cm.jet,
]

twitter = Twython(API_KEY, API_SECRET_KEY, ACCESS_TOKEN, ACCESS_SECRET_TOKEN)


def update_status(filename=None):
    """Animate

    Args:
        save_movie : boolean, default=True

    Returns:
        None
    """
    if filename is None:
        pendulum = Animation()
        pendulum.animate()
        print(pendulum.fname)


def media(filename=None, number=1, clean=False):
    """Animate

    Args:
        save_movie : boolean, default=True

    Returns:
        None
    """
    if filename is None and number <= 1:
        pendulum = Animation(fps=60, tend=10.0)
        pendulum.main_animate()
        status = (
            "Starting angles {} degrees and {} degrees\n\n#science #physics".format(
                np.round(np.rad2deg(pendulum.alpha), 2),
                np.round(np.rad2deg(pendulum.beta), 2),
            )
        )
        filename = pendulum.filename

    elif filename is None and number > 1:
        cmap = np.random.choice(cmaps)
        offset = np.random.uniform(1e-5, 1e-4)
        pendulum = Animation(fps=60, tend=10.0, npends=number, offset=offset)
        status = (
            "6 double pendulums separated by {} degrees\n\n#science #physics".format(
                np.round(np.rad2deg(offset), 4)
            )
        )
        pendulum.main_n_animate(cmap=cmap)
        filename = pendulum.filename

    video = open(filename, "rb")
    response = twitter.upload_video(media=video, media_type="video/gif")

    if clean:
        os.remove(filename)

    return status, response


def post_content(clean=False):
    """ """
    npend = np.random.choice([1, 6], p=[0.25, 0.75])
    status, response = media(number=npend, clean=clean)
    twitter.update_status(status=status, media_ids=[response["media_id"]])
