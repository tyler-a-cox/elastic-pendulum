import os
import tweepy
from twython import Twython
from .animate import Animation
from .double_pendulum import ElasticPendulum
from ._keys import API_KEY, API_SECRET_KEY, ACCESS_TOKEN, ACCESS_SECRET_TOKEN

twitter = Twython(API_KEY, API_SECRET_KEY, ACCESS_TOKEN, ACCESS_SECRET_TOKEN)

# auth = tweepy.OAuthHandler(API_KEY, API_SECRET_KEY)
# auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET_TOKEN)

# api = tweepy.API(auth)


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


def media(filename=None, number=1):
    """Animate

    Args:
        save_movie : boolean, default=True

    Returns:
        None
    """
    if filename is None and number <= 1:
        pendulum = ElasticPendulum(fps=24, t_end=15.0)
        _ = pendulum.integrate()
        pendulum.animate_spring()
        filename = pendulum.fname

    elif filename is None and number > 1:
        pendulum = Animation()
        pendulum.animate()
        filename = pendulum.fname

    video = open(filename, "rb")
    response = twitter.upload_video(media=video, media_type="video/mp4")
    return response
