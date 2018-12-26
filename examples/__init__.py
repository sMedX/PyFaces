__author__ = 'Ruslan N. Kosarev'

import os


# ======================================================================================================================
# directories
class dirs:
    inpdir = os.path.join(os.path.pardir, 'data')
    outdir = os.path.join(os.path.pardir, 'output')


# models
class models:
    bfm2017nomouth = os.path.join(dirs.inpdir, 'model2017-1_bfm_nomouth.h5')
    bfm2017face12nomouth = os.path.join(dirs.inpdir, 'model2017-1_face12_nomouth.h5')
# ======================================================================================================================
