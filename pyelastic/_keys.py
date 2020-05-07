import os

# Twitter API
API_KEY = 'eIh1gPxjWbbadGr1QBcqbq9QH'
API_SECRET_KEY = 'eAcgc79yWQiMxYFiUTAJKLjO0htfanE6SE1wgkLe5RG1g3559t'
ACCESS_TOKEN = '1252326049142075392-cEyVXr7YEd9kApuaaHndDBFDJSpRHC'
ACCESS_SECRET_TOKEN = 'fwt9Fn68JKe4lb1KCzqYlEGO4tNb8We2sM3sNfkIRjDcw'

# Save Directories
FIG_DIR = os.path.join('/Users/tyler/', '_figs')
VID_DIR = os.path.join('/Users/tyler/', '_videos')

if not os.path.exists(FIG_DIR):
    os.mkdir(FIG_DIR)

if not os.path.exists(VID_DIR):
    os.mkdir(VID_DIR)
