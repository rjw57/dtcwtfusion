# vim: set fileencoding=utf8 :

"""Convert output from fusevideo to MJPEG files.

Usage:
    hdf5toavi [options] <hdf5> <avi>
    hdf5toavi (-h | --help)

Options:
    -v, --verbose                       Increase logging verbosity.
    --write-input=FILE                  Save input frames to FILE

"""

from __future__ import print_function, division, absolute_import

import logging
import sys

from docopt import docopt
from six.moves import xrange
import h5py
import cv2
import numpy as np

def tonemap(array):
    # The normalisation strategy here is to let the middle 98% of
    # the values fall in the range 0.01 to 0.99 ('black' and 'white' level).
    black_level = np.percentile(array,  1)
    white_level = np.percentile(array, 99)

    norm_array = array - black_level
    norm_array /= (white_level - black_level)
    norm_array = np.clip(norm_array + 0.01, 0, 1)

    return np.array(norm_array * 255, dtype=np.uint8)

def main():
    options = docopt(__doc__)

    # Set up logging according to command line options
    loglevel = logging.INFO if options['--verbose'] else logging.WARN
    logging.basicConfig(level=loglevel)

    logging.info('Opening "{0}"'.format(options['<hdf5>']))
    h5 = h5py.File(options['<hdf5>'])

    if options['--write-input'] is not None:
        logging.info('Writing input frames to "{0}"'.format(options['--write-input']))

        vw = cv2.VideoWriter(options['--write-input'], cv2.cv.FOURCC(*'MJPG'),
                10, h5['input'].shape[1::-1], False)
        for idx in xrange(h5['input'].shape[2]):
            vw.write(tonemap(h5['input'][:,:,idx]))
        vw.release()

    logging.info('Writing fused and denoised frames to "{0}"'.format(options['<avi>']))
    vw = cv2.VideoWriter(options['<avi>'], cv2.cv.FOURCC(*'MJPG'),
            10, h5['denoised'].shape[1::-1], False)
    for idx in xrange(h5['denoised'].attrs['frame_count']):
        vw.write(tonemap(h5['denoised'][:,:,idx]))
    vw.release()
