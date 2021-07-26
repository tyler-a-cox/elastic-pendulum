import os
import sys
import numpy as np
import cmasher as cm
import pylab as plt
from twython import Twython
from .animate import Animation
from .settings import COLORMAPS

try:
    from ._keys import API_KEY, API_SECRET_KEY, ACCESS_TOKEN, ACCESS_SECRET_TOKEN

except ImportError:
    API_KEY = sys.getenv("API_KEY")
    API_SECRET_KEY = sys.getenv("API_SECRET_KEY")
    ACCESS_TOKEN = sys.getenv("ACCESS_TOKEN")
    ACCESS_SECRET_TOKEN = sys.getenv("ACCESS_SECRET_TOKEN")


hasttags = ["#science", "#scicomm", "#physics", "#sciart"]
tags = " ".join(hasttags)

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
        pendulum = Animation(fps=60, tend=20.0)
        pendulum.main_animate()
        status = "Starting angles {} degrees and {} degrees\n\n".format(
            np.round(np.rad2deg(pendulum.alpha), 2),
            np.round(np.rad2deg(pendulum.beta[0]), 2),
        )
        status += tags
        filename = pendulum.filename

    elif filename is None and number > 1:
        cmap = np.random.choice(COLORMAPS)
        offset = np.random.uniform(1e-3, 0.1)
        pendulum = Animation(fps=60, tend=20.0, npends=number, offset=offset)
        status = "6 indentical double pendulums separated by {} degrees\n\n".format(
            np.round(np.rad2deg(offset), 4)
        )
        status += tags
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
