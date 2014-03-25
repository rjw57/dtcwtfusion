# vim: set fileencoding=utf8 :

"""Convert output from fusevideo to MJPEG files.

Usage:
    hdf5toavi [options] <hdf5> <avi>
    hdf5toavi (-h | --help)

Options:
    -v, --verbose                       Increase logging verbosity.
    --write-input=FILE                  Save input frames to FILE
    -c, --comparison                    Generate side-by-side comparison
    --fps=NUM                           Set frames per second. [default: 10]

"""

from __future__ import print_function, division, absolute_import

import itertools
import logging
import sys

from docopt import docopt
from six.moves import xrange
import h5py
import cv2
import numpy as np

def tonemap(array):
    # The normalisation strategy here is to let the middle 95% of
    # the values fall in the range 0.025 to 0.975 ('black' and 'white' level).
    black_level = np.percentile(array,  2.5)
    white_level = np.percentile(array, 97.5)

    norm_array = array - black_level
    norm_array *= 0.95 / (white_level - black_level)
    norm_array = np.clip(norm_array + 0.025, 0, 1)

    return np.array(norm_array * 255, dtype=np.uint8)

def write_output(filename, fps, left, right=None, leftidxs=None, rightidxs=None):
    if right is None:
        output_shape = left.shape[:2]
        n_frames = left.shape[2]
    else:
        output_shape = (
                max(left.shape[0], right.shape[0]),
                left.shape[1] + right.shape[1]
        )
        n_frames = min(left.shape[2], right.shape[2])

    output_frame = np.zeros(output_shape, dtype=np.uint8)

    if leftidxs is None:
        leftidxs = np.arange(left.shape[2])
    if rightidxs is None:
        if right is not None:
            rightidxs = np.arange(right.shape[2])
        else:
            rightidxs = leftidxs

    vw = cv2.VideoWriter(filename, cv2.cv.FOURCC(*'MJPG'),
            fps, output_frame.shape[::-1], False)

    for lidx, ridx in itertools.izip(leftidxs, rightidxs):
        output_frame[:left.shape[0], :left.shape[1]] = tonemap(left[:,:,lidx])

        if right is not None:
            output_frame[:right.shape[0], left.shape[1]:(left.shape[1]+right.shape[1])] = \
                    tonemap(right[:,:,ridx])

        vw.write(output_frame)

    vw.release()

def main():
    options = docopt(__doc__)
    fps = int(options['--fps'])

    # Set up logging according to command line options
    loglevel = logging.INFO if options['--verbose'] else logging.WARN
    logging.basicConfig(level=loglevel)

    logging.info('Opening "{0}"'.format(options['<hdf5>']))
    h5 = h5py.File(options['<hdf5>'])

    input_frames = h5['input']
    denoised_frames = h5['denoised']

    if options['--write-input'] is not None:
        logging.info('Writing input frames to "{0}"'.format(options['--write-input']))

        write_output(options['--write-input'], fps, input_frames)

        #vw = cv2.VideoWriter(options['--write-input'], cv2.cv.FOURCC(*'MJPG'),
        #        fps, input_frames.shape[1::-1], False)
        #for idx in xrange(input_frames.shape[2]):
        #    vw.write(tonemap(input_frames[:,:,idx]))
        #vw.release()

    logging.info('Writing fused and denoised frames to "{0}"'.format(options['<avi>']))
    if options['--comparison']:
        write_output(options['<avi>'], fps,
                left=input_frames, right=denoised_frames,
                leftidxs=h5['processed_indices'])
    else:
        write_output(options['<avi>'], fps, denoised_frames)
    #vw = cv2.VideoWriter(options['<avi>'], cv2.cv.FOURCC(*'MJPG'),
    #        fps, denoised_frames.shape[1::-1], False)
    #for idx in xrange(denoised_frames.attrs['frame_count']):
    #    vw.write(tonemap(denoised_frames[:,:,idx]))
    #vw.release()
