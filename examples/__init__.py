__author__ = 'Ruslan N. Kosarev'

import os


# ======================================================================================================================
# directories and functions to join input and output directories
class dirs:
    inpdir = os.path.join(os.path.pardir, 'data')
    outdir = os.path.join(os.path.pardir, 'output')


def joininpdir(filename):
    return os.path.join(dirs.inpdir, filename)


def joinoutdir(filename):
    return os.path.join(dirs.outdir, filename)


# ======================================================================================================================
# models
class models:
    bfm2017nomouth = joininpdir('model2017-1_bfm_nomouth.h5')
    bfm2017face12nomouth = joininpdir('model2017-1_face12_nomouth.h5')


# ======================================================================================================================


