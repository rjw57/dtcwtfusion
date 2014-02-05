"""Align, register and fuse images specified on the command line using the
DTCWT fusion algorithm.

Usage:
    fuseimages [options] <images>...
    fuseimages (-h | --help)

Options:

    -n, --normalise                     Re-normalise input images to the interval [0,1].
    -v, --verbose                       Increase logging verbosity.
    -o PREFIX, --output-prefix=PREFIX   Prefix output filenames with PREFIX.
                                        [default: fused-]

"""

from __future__ import print_function, division

import logging

import dtcwt.backend.backend_numpy as dtcwtbackend
import dtcwt.registration
import dtcwt.sampling
from docopt import docopt
import numpy as np
from PIL import Image
from scipy.signal import fftconvolve
from six.moves import xrange

def load_images(filenames, normalise=False):
    """Load images from *filenames* returning a 3D-array with all pixel values.
    If *normalise* is True, then input images are normalised onto the range
    [0,1].

    """
    images = []

    for fn in filenames:
        # Load image with PIL
        logging.info('Loading "{fn}"'.format(fn=fn))
        im = Image.open(fn)

        # Extract data and store as floating point data
        im_array = np.array(im.getdata(), dtype=np.float32).reshape(im.size[::-1])

        # If we are to normalise, do so
        if normalise:
            im_array -= im_array.min()
            im_array /= im_array.max()

        # If this isn't the first image, check that shapes are consistent
        if len(images) > 0:
            if im_array.shape != images[-1].shape:
                logging.warn('Skipping "{fn}" with inconsistent shape'.format(fn=fn))
                continue

        images.append(im_array)

    logging.info('Loaded {0} image(s)'.format(len(images)))

    return np.dstack(images)

def align(frames, template):
    """
    Warp each slice of the 3D array frames to align it to *template*.

    """
    if frames.shape[:2] != template.shape:
        raise ValueError('Template must be same shape as one slice of frame array')

    # Calculate xs and ys to sample from one frame
    xs, ys = np.meshgrid(np.arange(frames.shape[1]), np.arange(frames.shape[0]))

    # Calculate window to use in FFT convolve
    w = np.outer(np.hanning(template.shape[0]), np.hanning(template.shape[1]))

    # Normalise template
    tmpl_min = template.min()
    norm_template = template - tmpl_min
    tmpl_max = norm_template.max()
    norm_template /= tmpl_max

    warped_ims = []
    for frame_idx in xrange(frames.shape[2]):
        logging.info('Aligning frame {0}/{1}'.format(frame_idx+1, frames.shape[2]))
        frame = frames[:,:,frame_idx]

        # Normalise frame
        norm_frame = frame - tmpl_min
        norm_frame /= tmpl_max

        # Convolve template and frame
        conv_im = fftconvolve(norm_template, np.fliplr(np.flipud(norm_frame)))

        # Find maximum location
        max_loc = np.unravel_index(conv_im.argmax(), conv_im.shape)

        # Convert location to shift
        dy = max_loc[0] - template.shape[0]
        dx = max_loc[1] - template.shape[1]

        # Warp image
        warped_ims.append(dtcwt.sampling.sample(frame, xs-dx, ys-dy, method='bilinear'))

    return np.dstack(warped_ims)

def register(frames, template, nlevels=7):
    """
    Use DTCWT registration to return warped versions of frames aligned to template.

    """
    # Normalise template
    tmpl_min = template.min()
    norm_template = template - tmpl_min
    tmpl_max = norm_template.max()
    norm_template /= tmpl_max

    # Transform template
    transform = dtcwtbackend.Transform2d()
    template_t = transform.forward(norm_template, nlevels=nlevels)

    warped_ims = []
    for frame_idx in xrange(frames.shape[2]):
        logging.info('Registering frame {0}/{1}'.format(frame_idx+1, frames.shape[2]))
        frame = frames[:,:,frame_idx]

        # Normalise frame
        norm_frame = frame - tmpl_min
        norm_frame /= tmpl_max

        # Transform frame
        frame_t = transform.forward(norm_frame, nlevels=nlevels)

        # Register
        reg = dtcwt.registration.estimatereg(frame_t, template_t)
        warped_ims.append(dtcwt.registration.warp(frame, reg, method='bilinear'))

    return np.dstack(warped_ims)

def save_image(filename, array):
    # The normalisation strategy here is to let the middle 98% of
    # the values fall in the range 0.01 to 0.99 ('black' and 'white' level).
    black_level = np.percentile(array,  1)
    white_level = np.percentile(array, 99)

    norm_array = array - black_level
    norm_array /= (white_level - black_level)
    norm_array = np.clip(norm_array + 0.01, 0, 1)

    # Copy is workaround for http://goo.gl/8fuOJA
    im_array = np.array(norm_array * 255, dtype=np.uint8)
    im = Image.fromarray(im_array.copy(), 'L')

    logging.info('Saving "{0}"'.format(filename + '.png'))
    im.save(filename + '.png')

def transform_frames(frames, nlevels=7):
    # Transform each registered frame storing result
    lowpasses = []
    highpasses = []
    for idx in xrange(nlevels):
        highpasses.append([])

    transform = dtcwtbackend.Transform2d()
    for frame_idx in xrange(frames.shape[2]):
        logging.info('Transforming frame {0}/{1}'.format(frame_idx+1, frames.shape[2]))
        frame = frames[:,:,frame_idx]
        frame_t = transform.forward(frame, nlevels=nlevels)

        lowpasses.append(frame_t.lowpass)
        for idx in xrange(nlevels):
            highpasses[idx].append(frame_t.subbands[idx][:,:,:,np.newaxis])

    return np.dstack(lowpasses), tuple(np.concatenate(hp, axis=3) for hp in highpasses)

def reconstruct(lowpass, highpasses):
    transform = dtcwtbackend.Transform2d()
    t = dtcwtbackend.TransformDomainSignal(lowpass, highpasses)
    return transform.inverse(t).value

def main():
    options = docopt(__doc__)
    imprefix = options['--output-prefix']

    # Set up logging according to command line options
    loglevel = logging.INFO if options['--verbose'] else logging.WARN
    logging.basicConfig(level=loglevel)

    # Load inputs
    logging.info('Loading input images')
    input_frames = load_images(options['<images>'], options['--normalise'])

    # Save sample frame
    logging.info('Saving sample frame')
    save_image(imprefix + 'sample-frame', input_frames[:,:,input_frames.shape[2]>>1])

    # Align images to *centre* frame
    logging.info('Aligning images')
    aligned_frames = align(input_frames, input_frames[:,:,input_frames.shape[2]>>1])

    # Save mean aligned frame
    logging.info('Saving mean aligned frame')
    save_image(imprefix + 'mean-aligned', np.mean(aligned_frames, axis=2))

    # Register images to mean aligned frame
    registered_frames = register(aligned_frames, np.mean(aligned_frames, axis=2))

    # Save mean registered frame
    logging.info('Saving mean registered frame')
    save_image(imprefix + 'mean-registered', np.mean(registered_frames, axis=2))

    logging.info('Saving registered frames')
    np.savez_compressed(imprefix + 'registered-frames.npz', frames=registered_frames)

    # Transform registered frames
    lowpasses, highpasses = transform_frames(registered_frames)

    # Compute mean lowpass image
    lowpass_mean = np.mean(lowpasses, axis=2)

    # Get mean direction for each subband
    phases = []
    for level_sb in highpasses:
        # Calculate mean direction by adding all subbands together and normalising
        sum_ = np.sum(level_sb, axis=3)
        sum_mag = np.abs(sum_)
        sum_ /= np.where(sum_mag != 0, sum_mag, 1)
        phases.append(sum_)

    # Compute mean, maximum and maximum-of-inliers magnitudes
    mean_mags, max_mags, max_inlier_mags = [], [], []
    for level_sb in highpasses:
        mags = np.abs(level_sb)

        mean_mags.append(np.mean(mags, axis=3))
        max_mags.append(np.max(mags, axis=3))

        thresh = 2*np.repeat(np.median(mags, axis=3)[:,:,:,np.newaxis], level_sb.shape[3], axis=3)
        outlier_suppressed = np.where(mags < thresh, mags, 0)
        max_inlier_mags.append(np.max(outlier_suppressed, axis=3))

    # Reconstruct frames
    logging.info('Computing mean magnitude fused image')
    mean_recon = reconstruct(lowpass_mean, tuple(mag*phase for mag, phase in zip(mean_mags, phases)))
    save_image(imprefix + 'fused-mean-dtcwt', mean_recon)

    logging.info('Computing maximum magnitude fused image')
    max_recon = reconstruct(lowpass_mean, tuple(mag*phase for mag, phase in zip(max_mags, phases)))
    save_image(imprefix + 'fused-max-dtcwt', max_recon)

    logging.info('Computing maximum of inliners magnitude fused image')
    max_inlier_recon = reconstruct(lowpass_mean, tuple(mag*phase for mag, phase in zip(max_inlier_mags, phases)))
    save_image(imprefix + 'fused-max-inlier-dtcwt', max_inlier_recon)

