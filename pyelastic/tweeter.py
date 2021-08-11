import os
import sys
import numpy as np
import cmasher as cm
import pylab as plt
from twython import Twython
from .animate import Animation
from .settings import COLORMAPS, COLORS, tags

try:
    from ._keys import API_KEY, API_SECRET_KEY, ACCESS_TOKEN, ACCESS_SECRET_TOKEN

except ImportError:
    API_KEY = os.getenv("API_KEY")
    API_SECRET_KEY = os.getenv("API_SECRET")
    ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
    ACCESS_SECRET_TOKEN = os.getenv("ACCESS_TOKEN_SECRET")

# Authorization through Twython
twitter = Twython(API_KEY, API_SECRET_KEY, ACCESS_TOKEN, ACCESS_SECRET_TOKEN)


def media(number=1, clean=False):
    """Setup the simulations for posting

    Args:
        number : int, default=1
            Number of pendulums to simulate
        clean : bool, default=False
            If true, the video will be deleted after being posted

    Returns:
        status : str
            Text to be posted with the video describing the video
        response : dict
            Dictionary describing the metadata of the tweet

    """
    if number == 1:
        i, j = i, j = np.random.choice(np.arange(len(COLORS)), replace=False, size=2)
        pendulum = Animation(fps=60, tend=20.0)
        pendulum.animate(colors=(COLORS[i], COLORS[j]))
        status = "Starting angles {} degrees and {} degrees\n\n".format(
            np.round(np.rad2deg(pendulum.alpha), 2),
            np.round(np.rad2deg(pendulum.beta[0]), 2),
        )
        status += tags
        filename = pendulum.filename

    else:
        cmap = np.random.choice(COLORMAPS)
        offset = np.random.uniform(1e-3, 0.1)
        pendulum = Animation(fps=60, tend=20.0, npends=number, offset=offset)
        status = "6 indentical double pendulums separated by {} degrees\n\n".format(
            np.round(np.rad2deg(offset), 4)
        )
        status += tags
        pendulum.animate(cmap=cmap)
        filename = pendulum.filename

    video = open(filename, "rb")
    response = twitter.upload_video(media=video, media_type="video/gif")

    if clean:
        os.remove(filename)

    return status, response


def tweet(clean=False):
    """Tweet out a video of a springy double pendulum

    Args:
        clean : bool, default=False
            If true, the video will be deleted after being posted
    """
    npend = np.random.choice([1, 6], p=[0.4, 0.6])
    status, response = media(number=npend, clean=clean)
    twitter.update_status(status=status, media_ids=[response["media_id"]])
