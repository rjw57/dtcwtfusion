# vim: set fileencoding=utf8 :

"""Align, register and fuse frames specified on the command line using the
DTCWT fusion algorithm.

Usage:
    fusevideo [options] <frames>...
    fusevideo (-h | --help)

Options:
    -n, --normalise                     Re-normalise input frames to lie on the
                                        interval [0,1].
    -v, --verbose                       Increase logging verbosity.
    -o PREFIX, --output-prefix=PREFIX   Prefix output filenames with PREFIX.
                                        [default: fused-]
    -w FRAMES, --window=FRAMES          Sliding half-window size. [default: 5]

"""

from __future__ import print_function, division, absolute_import

import logging

from ._images2gif import writeGif

import dtcwt
import dtcwt.registration
import dtcwt.sampling
from docopt import docopt
import numpy as np
from PIL import Image
from scipy.signal import fftconvolve
from six.moves import xrange

def load_frames(filenames, normalise=False):
    """Load frames from *filenames* returning a 3D-array with all pixel values.
    If *normalise* is True, then input frames are normalised onto the range
    [0,1].

    """
    frames = []

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
        if len(frames) > 0:
            if im_array.shape != frames[-1].shape:
                logging.warn('Skipping "{fn}" with inconsistent shape'.format(fn=fn))
                continue

        frames.append(im_array)

    logging.info('Loaded {0} image(s)'.format(len(frames)))

    return np.dstack(frames)

def align(frames, template):
    """
    Warp each slice of the 3D array frames to align it to *template*.

    """
    if frames.shape[:2] != template.shape:
        raise ValueError('Template must be same shape as one slice of frame array')

    # Calculate xs and ys to sample from one frame
    xs, ys = np.meshgrid(np.arange(frames.shape[1]), np.arange(frames.shape[0]))

    # Calculate window to use in FFT convolve
    w = np.outer(np.hamming(template.shape[0]), np.hamming(template.shape[1]))

    # Calculate a normalisation for the cross-correlation
    ccnorm = 1.0 / fftconvolve(w, w)

    # Set border of normalisation to zero to avoid overfitting. Borser is set so that there
    # must be a minimum of half-frame overlap
    ccnorm[:(template.shape[0]>>1),:] = 0
    ccnorm[-(template.shape[0]>>1):,:] = 0
    ccnorm[:,:(template.shape[1]>>1)] = 0
    ccnorm[:,-(template.shape[1]>>1):] = 0

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
        conv_im = fftconvolve(norm_template*w, np.fliplr(np.flipud(norm_frame*w)))
        conv_im *= ccnorm

        # Find maximum location
        max_loc = np.unravel_index(conv_im.argmax(), conv_im.shape)

        # Convert location to shift
        dy = max_loc[0] - template.shape[0] + 1
        dx = max_loc[1] - template.shape[1] + 1
        logging.info('Offset computed to be ({0},{1})'.format(dx, dy))

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
    transform = dtcwt.Transform2d()
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

def tonemap(array):
    # The normalisation strategy here is to let the middle 98% of
    # the values fall in the range 0.01 to 0.99 ('black' and 'white' level).
    black_level = np.percentile(array,  1)
    white_level = np.percentile(array, 99)

    norm_array = array - black_level
    norm_array /= (white_level - black_level)
    norm_array = np.clip(norm_array + 0.01, 0, 1)

    return np.array(norm_array * 255, dtype=np.uint8)

def save_image(filename, array):
    # Copy is workaround for http://goo.gl/8fuOJA
    im = Image.fromarray(tonemap(array).copy(), 'L')

    logging.info('Saving "{0}"'.format(filename + '.png'))
    im.save(filename + '.png')

def transform_frames(frames, nlevels=7):
    # Transform each registered frame storing result
    lowpasses = []
    highpasses = []
    for idx in xrange(nlevels):
        highpasses.append([])

    transform = dtcwt.Transform2d()
    for frame_idx in xrange(frames.shape[2]):
        logging.info('Transforming frame {0}/{1}'.format(frame_idx+1, frames.shape[2]))
        frame = frames[:,:,frame_idx]
        frame_t = transform.forward(frame, nlevels=nlevels)

        lowpasses.append(frame_t.lowpass)
        for idx in xrange(nlevels):
            highpasses[idx].append(frame_t.highpasses[idx][:,:,:,np.newaxis])

    return np.dstack(lowpasses), tuple(np.concatenate(hp, axis=3) for hp in highpasses)

def reconstruct(lowpass, highpasses):
    transform = dtcwt.Transform2d()
    t = dtcwt.Pyramid(lowpass, highpasses)
    return transform.inverse(t)

def shrink_coeffs(highpasses):
    """Implement Bivariate Laplacian shrinkage as described in [1].
    *highpasses* is a sequence containing wavelet coefficients for each level
    fine-to-coarse. Return a sequence containing the shrunken coefficients.

    [1] A. Loza, D. Bull, N. Canagarajah, and A. Achim, “Non-gaussian model-
    based fusion of noisy frames in the wavelet domain,” Comput. Vis. Image
    Underst., vol. 114, pp. 54–65, Jan. 2010.

    """
    shrunk_levels = []

    # Estimate noise from first level coefficients:
    # \sigma_n = MAD(X_1) / 0.6745

    # Compute median absolute deviation of wavelet magnitudes. This is more than
    # a little magic compared to the 1d version.
    level1_mad_real = np.median(np.abs(highpasses[0].real - np.median(highpasses[0].real)))
    level1_mad_imag = np.median(np.abs(highpasses[0].imag - np.median(highpasses[0].imag)))
    sigma_n = np.sqrt(level1_mad_real*level1_mad_real + level1_mad_imag+level1_mad_imag) / (np.sqrt(2) * 0.6745)

    # In this context, parent == coarse, child == fine. Work from
    # coarse to fine
    shrunk_levels.append(highpasses[-1])
    for parent, child in zip(highpasses[-1:0:-1], highpasses[-2::-1]):
        # We will shrink child coefficients.

        # Rescale parent to be the size of child
        parent = dtcwt.sampling.rescale(parent, child.shape[:2], method='nearest')

        # Construct gain for shrinkage separately per direction and for real and imag
        real_gain = np.ones_like(child.real)
        imag_gain = np.ones_like(child.real)
        for dir_idx in xrange(parent.shape[2]):
            child_d = child[:,:,dir_idx]
            parent_d = parent[:,:,dir_idx]

            # Estimate sigma_w and gain for real
            real_sigma_w = np.sqrt(np.maximum(1e-8, np.var(child_d.real) - sigma_n*sigma_n))
            real_R = np.sqrt(parent_d.real*parent_d.real + child_d.real*child_d.real)
            real_gain[:,:,dir_idx] = np.maximum(0, real_R - (np.sqrt(3)*sigma_n*sigma_n)/real_sigma_w) / real_R

            # Estimate sigma_w and gain for imag
            imag_sigma_w = np.sqrt(np.maximum(1e-8, np.var(child_d.imag) - sigma_n*sigma_n))
            imag_R = np.sqrt(parent_d.imag*parent_d.imag + child_d.imag*child_d.imag)
            imag_gain[:,:,dir_idx] = np.maximum(0, imag_R - (np.sqrt(3)*sigma_n*sigma_n)/imag_sigma_w) / imag_R

        # Shrink child levels
        shrunk = (child.real * real_gain) + 1j * (child.imag * imag_gain)
        shrunk_levels.append(shrunk)

    return shrunk_levels[::-1]

def main():
    options = docopt(__doc__)
    imprefix = options['--output-prefix']

    # Set up logging according to command line options
    loglevel = logging.INFO if options['--verbose'] else logging.WARN
    logging.basicConfig(level=loglevel)

    # Load inputs
    logging.info('Loading input frames')
    input_frames = load_frames(options['<frames>'], options['--normalise'])

    # Select reference frames according to window
    frame_indices = np.arange(input_frames.shape[2])
    half_window_size = int(options['--window'])
    for ref_idx in frame_indices[half_window_size:-(half_window_size+1)]:
        logging.info('Processing frame {0}'.format(ref_idx))

        reference_frame = input_frames[:,:,ref_idx]

        logging.info('Saving input frame')
        save_image(imprefix + 'input-{0:05d}'.format(ref_idx-half_window_size),
                reference_frame)

        stack = input_frames[:,:,ref_idx-half_window_size:ref_idx+half_window_size+1]

        # Align frames to *centre* frame
        logging.info('Aligning frames')
        aligned_frames = align(stack, reference_frame)

        # Register frames
        logging.info('Registering frames')
        registration_reference = reference_frame
        registered_frames = register(aligned_frames, registration_reference)

        # Save mean registered frame
        logging.info('Saving mean registered frame')
        save_image(imprefix + 'mean-registered-{0:05d}'.format(ref_idx-half_window_size),
                np.mean(registered_frames, axis=2))

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
        logging.info('Computing maximum of inliners magnitude fused image')
        max_inlier_recon = reconstruct(lowpass_mean,
                tuple(mag*phase for mag, phase in zip(max_inlier_mags, phases)))
        save_image(imprefix + 'fused-max-inlier-dtcwt-{0:05d}'.format(ref_idx-half_window_size),
                max_inlier_recon)

        logging.info('Computing maximum of inliners magnitude fused image w/ shrinkage')
        max_inlier_shrink_recon = reconstruct(lowpass_mean,
                shrink_coeffs(tuple(mag*phase for mag, phase in zip(max_inlier_mags, phases))))
        save_image(imprefix + 'fused-max-inlier-shrink-{0:05d}'.format(ref_idx-half_window_size),
                max_inlier_shrink_recon)