"""
Stereo rectification tools
Copyright (C) 2018, Gabriele Facciolo <facciolo@cmla.ens-cachan.fr>
Copyright (C) 2018, Carlo de Franchis <carlo.de-franchis@ens-cachan.fr>
"""

from __future__ import print_function
from scipy import ndimage
import numpy as np
import cv2
import ad

import libs2p.utils as utils
import srtm4

import os
import rpcm

import libs2p.rpc_utils
import libs2p.estimation
import libs2p.evaluation
import libs2p.common
import libs2p.visualisation
import libs2p.block_matching
from libs2p.config import cfg

def disparity_range_from_matches(matches, H1, H2, w, h):
    """
    Compute the disparity range of a ROI from a list of point matches.

    Args:
        matches: Nx4 numpy array containing a list of matches, in the full
            image coordinates frame, before rectification
        w, h: width and height of the rectangular ROI in the first image.
        H1, H2: two rectifying homographies, stored as numpy 3x3 matrices

    Returns:
        disp_min, disp_max: horizontal disparity range
    """
    # transform the matches according to the homographies
    p1 = libs2p.common.points_apply_homography(H1, matches[:, :2])
    x1 = p1[:, 0]
    p2 = libs2p.common.points_apply_homography(H2, matches[:, 2:])
    x2 = p2[:, 0]

    # compute the final disparity range
    disp_min = np.floor(np.min(x2 - x1))
    disp_max = np.ceil(np.max(x2 - x1))

    # add a security margin to the disparity range
    disp_min -= (disp_max - disp_min) * cfg['disp_range_extra_margin']
    disp_max += (disp_max - disp_min) * cfg['disp_range_extra_margin']
    return disp_min, disp_max


def disparity_range(rpc1, rpc2, x, y, w, h, H1, H2, matches, A=None):
    """
    Compute the disparity range of a ROI from a list of point matches.

    Args:
        rpc1, rpc2 (rpcm.RPCModel): two RPC camera models
        x, y, w, h (int): 4-tuple of integers defining the rectangular ROI in
            the first image. (x, y) is the top-left corner, and (w, h) are the
            dimensions of the rectangle.
        H1, H2 (np.array): two rectifying homographies, stored as 3x3 arrays
        matches (np.array): Nx4 array containing a list of sift matches, in the
            full image coordinates frame
        A (np.array): 3x3 array containing the pointing error correction for
            im2. This matrix is usually estimated with the pointing_accuracy
            module.

    Returns:
        disp: 2-uple containing the horizontal disparity range
    """
    # compute exogenous disparity range if needed
    if cfg['disp_range_method'] in ['exogenous', 'wider_sift_exogenous']:
        exogenous_disp = rpc_utils.exogenous_disp_range_estimation(rpc1, rpc2,
                                                                   x, y, w, h,
                                                                   H1, H2, A,
                                                                   cfg['disp_range_exogenous_high_margin'],
                                                                   cfg['disp_range_exogenous_low_margin'])

        #print("exogenous disparity range:", exogenous_disp)

    # compute SIFT disparity range if needed
    if cfg['disp_range_method'] in ['sift', 'wider_sift_exogenous']:
        if matches is not None and len(matches) >= 2:
            sift_disp = disparity_range_from_matches(matches, H1, H2, w, h)
        else:
            sift_disp = None
        #print("SIFT disparity range:", sift_disp)

    # compute altitude range disparity if needed
    if cfg['disp_range_method'] == 'fixed_altitude_range':
        alt_disp = rpc_utils.altitude_range_to_disp_range(cfg['alt_min'],
                                                          cfg['alt_max'],
                                                          rpc1, rpc2,
                                                          x, y, w, h, H1, H2, A)
        #print("disparity range computed from fixed altitude range:", alt_disp)

    # now compute disparity range according to selected method
    if cfg['disp_range_method'] == 'exogenous':
        disp = exogenous_disp

    elif cfg['disp_range_method'] == 'sift':
        disp = sift_disp

    elif cfg['disp_range_method'] == 'wider_sift_exogenous':
        if sift_disp is not None and exogenous_disp is not None:
            disp = min(exogenous_disp[0], sift_disp[0]), max(exogenous_disp[1], sift_disp[1])
        else:
            disp = sift_disp or exogenous_disp

    elif cfg['disp_range_method'] == 'fixed_altitude_range':
        disp = alt_disp

    elif cfg['disp_range_method'] == 'fixed_pixel_range':
        disp = cfg['disp_min'], cfg['disp_max']

    # default disparity range to return if everything else broke
    if disp is None:
        disp = -3, 3

    # impose a minimal disparity range (TODO this is valid only with the
    # 'center' flag for register_horizontally_translation)
    disp = min(-3, disp[0]), max(3, disp[1])

    #print("Final disparity range:", disp)
    return disp


def register_horizontally_translation(matches, H1, H2, flag='center'):
    """
    Adjust rectifying homographies with a translation to modify the disparity range.

    Args:
        matches: list of pairs of 2D points, stored as a Nx4 numpy array
        H1, H2: two homographies, stored as numpy 3x3 matrices
        flag: option needed to control how to modify the disparity range:
            'center': move the barycenter of disparities of matches to zero
            'positive': make all the disparities positive
            'negative': make all the disparities negative. Required for
                Hirshmuller stereo (java)

    Returns:
        H2: corrected homography H2

    The matches are provided in the original images coordinate system. By
    transforming these coordinates with the provided homographies, we obtain
    matches whose disparity is only along the x-axis. The second homography H2
    is corrected with a horizontal translation to obtain the desired property
    on the disparity range.
    """
    # transform the matches according to the homographies
    p1 = libs2p.common.points_apply_homography(H1, matches[:, :2])
    x1 = p1[:, 0]
    y1 = p1[:, 1]
    p2 = libs2p.common.points_apply_homography(H2, matches[:, 2:])
    x2 = p2[:, 0]
    y2 = p2[:, 1]

    # for debug, print the vertical disparities. Should be zero.
    if cfg['debug']:
        print("Residual vertical disparities: max, min, mean. Should be zero")
        print(np.max(y2 - y1), np.min(y2 - y1), np.mean(y2 - y1))

    # compute the disparity offset according to selected option
    t = 0
    if (flag == 'center'):
        t = np.mean(x2 - x1)
    if (flag == 'positive'):
        t = np.min(x2 - x1)
    if (flag == 'negative'):
        t = np.max(x2 - x1)

    # correct H2 with a translation
    return np.dot(libs2p.common.matrix_translation(-t, 0), H2)


def filter_matches_epipolar_constraint(F, matches, thresh):
    """
    Discards matches that are not consistent with the epipolar constraint.

    Args:
        F: fundamental matrix
        matches: list of pairs of 2D points, stored as a Nx4 numpy array
        thresh: maximum accepted distance between a point and its matched
            epipolar line

    Returns:
        the list of matches that satisfy the constraint. It is a sub-list of
        the input list.
    """
    out = []
    for match in matches:
        x = np.array([match[0], match[1], 1])
        xx = np.array([match[2], match[3], 1])
        d1 = libs2p.evaluation.distance_point_to_line(x, np.dot(F.T, xx))
        d2 = libs2p.evaluation.distance_point_to_line(xx, np.dot(F, x))
        if max(d1, d2) < thresh:
            out.append(match)

    return np.array(out)

def register_horizontally_shear(matches, H1, H2):
    """
    Adjust rectifying homographies with tilt, shear and translation to reduce the disparity range.

    Args:
        matches: list of pairs of 2D points, stored as a Nx4 numpy array
        H1, H2: two homographies, stored as numpy 3x3 matrices

    Returns:
        H2: corrected homography H2

    The matches are provided in the original images coordinate system. By
    transforming these coordinates with the provided homographies, we obtain
    matches whose disparity is only along the x-axis.
    """
    # transform the matches according to the homographies
    p1 = libs2p.common.points_apply_homography(H1, matches[:, :2])
    x1 = p1[:, 0]
    y1 = p1[:, 1]
    p2 = libs2p.common.points_apply_homography(H2, matches[:, 2:])
    x2 = p2[:, 0]
    y2 = p2[:, 1]

    if cfg['debug']:
        print("Residual vertical disparities: max, min, mean. Should be zero")
        print(np.max(y2 - y1), np.min(y2 - y1), np.mean(y2 - y1))

    # we search the (a, b, c) vector that minimises \sum (x1 - (a*x2+b*y2+c))^2
    # it is a least squares minimisation problem
    A = np.vstack((x2, y2, y2*0+1)).T
    a, b, c = np.linalg.lstsq(A, x1)[0].flatten()

    # correct H2 with the estimated tilt, shear and translation
    return np.dot(np.array([[a, b, c], [0, 1, 0], [0, 0, 1]]), H2)

def rectification_homographies(matches, x, y, w, h):
    """
    Computes rectifying homographies from point matches for a given ROI.

    The affine fundamental matrix F is estimated with the gold-standard
    algorithm, then two rectifying similarities (rotation, zoom, translation)
    are computed directly from F.

    Args:
        matches: numpy array of shape (n, 4) containing a list of 2D point
            correspondences between the two images.
        x, y, w, h: four integers defining the rectangular ROI in the first
            image. (x, y) is the top-left corner, and (w, h) are the dimensions
            of the rectangle.
    Returns:
        S1, S2, F: three numpy arrays of shape (3, 3) representing the
        two rectifying similarities to be applied to the two images and the
        corresponding affine fundamental matrix.
    """
    # estimate the affine fundamental matrix with the Gold standard algorithm
    F = libs2p.estimation.affine_fundamental_matrix(matches)

    # compute rectifying similarities
    S1, S2 = libs2p.estimation.rectifying_similarities_from_affine_fundamental_matrix(F, cfg['debug'])

    if cfg['debug']:
        y1 = libs2p.common.points_apply_homography(S1, matches[:, :2])[:, 1]
        y2 = libs2p.common.points_apply_homography(S2, matches[:, 2:])[:, 1]
        err = np.abs(y1 - y2)
        print("max, min, mean rectification error on point matches: ", end=' ')
        print(np.max(err), np.min(err), np.mean(err))

    # pull back top-left corner of the ROI to the origin (plus margin)
    pts = libs2p.common.points_apply_homography(S1, [[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
    x0, y0 = libs2p.common.bounding_box2D(pts)[:2]
    T = libs2p.common.matrix_translation(-x0, -y0)
    return np.dot(T, S1), np.dot(T, S2), F


def rectify_pair(im1, im2, rpc1, rpc2, x, y, w, h, out1, out2, A=None, sift_matches=None,
                 method='rpc', hmargin=0, vmargin=0):
    """
    Rectify a ROI in a pair of images.

    Args:
        im1, im2: paths to two GeoTIFF image files
        rpc1, rpc2: two instances of the rpcm.RPCModel class
        x, y, w, h: four integers defining the rectangular ROI in the first
            image.  (x, y) is the top-left corner, and (w, h) are the dimensions
            of the rectangle.
        out1, out2: paths to the output rectified crops
        A (optional): 3x3 numpy array containing the pointing error correction
            for im2. This matrix is usually estimated with the pointing_accuracy
            module.
        sift_matches (optional): Nx4 numpy array containing a list of sift
            matches, in the full image coordinates frame
        method (default: 'rpc'): option to decide wether to use rpc of sift
            matches for the fundamental matrix estimation.
        {h,v}margin (optional): horizontal and vertical margins added on the
            sides of the rectified images

    Returns:
        H1, H2: Two 3x3 matrices representing the rectifying homographies that
        have been applied to the two original (large) images.
        disp_min, disp_max: horizontal disparity range
    """
    # compute real or virtual matches
    if method == 'rpc':
        # find virtual matches from RPC camera models
        matches = libs2p.rpc_utils.matches_from_rpc(rpc1, rpc2, x, y, w, h,
                                             cfg['n_gcp_per_axis'])
        # correct second image coordinates with the pointing correction matrix
        if A is not None:
            matches[:, 2:] = libs2p.common.points_apply_homography(np.linalg.inv(A),
                                                            matches[:, 2:])
    elif method == 'sift':
        matches = sift_matches

    else:
        raise Exception("Unknown value {} for argument 'method'".format(method))

    # compute rectifying homographies
    H1, H2, F = rectification_homographies(matches, x, y, w, h)
   
    if cfg['register_with_shear']:
        # compose H2 with a horizontal shear to reduce the disparity range
        a = np.mean(libs2p.rpc_utils.altitude_range(rpc1, x, y, w, h))
        lon, lat, alt = libs2p.rpc_utils.ground_control_points(rpc1, x, y, w, h, a, a, 4)
        x1, y1 = rpc1.projection(lon, lat, alt)[:2]
        x2, y2 = rpc2.projection(lon, lat, alt)[:2]
        m = np.vstack([x1, y1, x2, y2]).T
        m = np.vstack({tuple(row) for row in m})  # remove duplicates due to no alt range
        H2 = register_horizontally_shear(m, H1, H2)

    # compose H2 with a horizontal translation to center disp range around 0
    if sift_matches is not None:
        sift_matches = filter_matches_epipolar_constraint(F, sift_matches,
                                                          cfg['epipolar_thresh'])
        if len(sift_matches) < 10:
            print('WARNING: no registration with less than 10 matches')
        else:
            H2 = register_horizontally_translation(sift_matches, H1, H2)

    # compute disparity range
    if cfg['debug']:
        out_dir = os.path.dirname(out1)
        np.savetxt(os.path.join(out_dir, 'sift_matches_disp.txt'),
                   sift_matches, fmt='%9.3f')
        visualisation.plot_matches(im1, im2, rpc1, rpc2, sift_matches, x, y, w, h,
                                   os.path.join(out_dir, 'sift_matches_disp.png'))
    
    # Compute the minimum and maximum disparities for the tile
    disp_m, disp_M = disparity_range(rpc1, rpc2, x, y, w, h, H1, H2,
                                     sift_matches, A)

    # recompute hmargin and homographies
    hmargin = int(np.ceil(max([hmargin, np.fabs(disp_m), np.fabs(disp_M)])))
    T = libs2p.common.matrix_translation(hmargin, vmargin)
    H1, H2 = np.dot(T, H1), np.dot(T, H2)
    
    # compute rectifying homographies for non-epipolar mode (rectify the secondary tile only)
    if libs2p.block_matching.rectify_secondary_tile_only(cfg['matching_algorithm']):
        H1_inv = np.linalg.inv(H1)
        H1 = np.eye(3) # H1 is replaced by 2-D array with ones on the diagonal and zeros elsewhere
        H2 = np.dot(H1_inv,H2)
        T = libs2p.common.matrix_translation(-x + hmargin, -y + vmargin)
        H1 = np.dot(T, H1)
        H2 = np.dot(T, H2)

    # compute output images size
    roi = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
    pts1 = libs2p.common.points_apply_homography(H1, roi)
    
    x0, y0, w0, h0 = libs2p.common.bounding_box2D(pts1)

    # check that the first homography maps the ROI in the positive quadrant
    np.testing.assert_allclose(np.round([x0, y0]), [hmargin, vmargin], atol=.01)

    # apply homographies and do the crops
    libs2p.common.image_apply_homography(out1, im1, H1, w0 + 2*hmargin, h0 + 2*vmargin)
    libs2p.common.image_apply_homography(out2, im2, H2, w0 + 2*hmargin, h0 + 2*vmargin)

    if libs2p.block_matching.rectify_secondary_tile_only(cfg['matching_algorithm']):
        pts_in = [[0, 0], [disp_m, 0], [disp_M, 0]]
        pts_out = libs2p.common.points_apply_homography(H1_inv,
                                                 pts_in)
        disp_m = pts_out[1,:] - pts_out[0,:]
        disp_M = pts_out[2,:] - pts_out[0,:]

    return H1, H2, disp_m, disp_M


def get_angle_from_cos_and_sin(c, s):
    """
    Computes x in ]-pi, pi] such that cos(x) = c and sin(x) = s.
    """
    if s >= 0:
        return np.arccos(c)
    else:
        return -np.arccos(c)


def matrix_translation(x, y):
    """
    Return the (3, 3) matrix representing a 2D shift in homogeneous coordinates.
    """
    t = np.eye(3)
    t[0, 2] = x
    t[1, 2] = y
    return t


def projection_of_3d_vertical_line(rpc, lon, lat, h_min=-200, h_max=3000, h_step=10):
    """
    Sample the projection of a 3d vertical line on the image plane.

    Args:
        rpc: instance of the rpc_model.RPCModel class
        lon, lat: geographic coordinates of the ground point through which the vertical line passes
        h_min, h_max: min, max altitude bounds of the vertical line
        h_step: step used to sample the vertical line

    Return:
        list of points in the image plane given by their pixel coordinates
    """
    return [rpc.projection(lon, lat, h) for h in np.arange(h_min, h_max, h_step)]


def epipolar_curve(rpc1, rpc2, x, y, h_min=-200, h_max=3000, h_step=10):
    """
    Sample the epipolar curve of image 2 associated to point (x, y) of image 1.

    Args:
        rpc1, rpc2: instances of the rpc_model.RPCModel class
        x, y: pixel coordinates of a point in the first image (ie associated to rpc1)
        h_min, h_max: min, max altitudes defining the bounds of the epipolar curve
        h_step: step used to sample the epipolar curve

    Return:
        list of points in the second image given by their pixel coordinates
    """
    return [rpc2.projection(*rpc1.localization(x, y, h), h) for h in
            np.arange(h_min, h_max, h_step)]


def trace_epipolar_curve(image1, image2, aoi, x0, y0):
    """auxiliary function to display in image2 the epipolar curve 
        corresponding to the point x0,y0 in the cropped image1"""
    
    import matplotlib.pyplot as plt

    # get the altitude of the center of the AOI
    lon, lat = aoi['center']
    z = srtm4.srtm4(lon, lat)

    # read the RPC coefficients of images i and j
    rpc1 = utils.rpc_from_geotiff(image1)
    rpc2 = utils.rpc_from_geotiff(image2)

    # crop the two images
    im1, x1, y1 = utils.crop_aoi(image1, aoi, z)
    im2, x2, y2 = utils.crop_aoi(image2, aoi, z)

    # translation matrices needed to compensate the crop offset
    H1 = matrix_translation(x1, y1)
    H2 = matrix_translation(x2, y2)

    # select a point in the first image
    #x0, y0 = 200, 200

    # compensate the crop offset of the first image
    x, y = np.dot(H1, [x0, y0, 1])[:2]

    # compute the epipolar curve
    epi = epipolar_curve(rpc1, rpc2, x, y)

    # compensate for the crop offset of the second image
    p = np.array([np.dot(np.linalg.inv(H2), [x, y, 1])[:2] for x, y in epi])

    # plot the epipolar curve on the second image
    f, ax = plt.subplots(1, 2, figsize=(13,10))
    ax[0].plot(x0, y0, 'r+')
    ax[1].plot(p[:, 0], p[:, 1], 'r-')
    ax[0].imshow(np.sqrt(im1.squeeze()), cmap='gray')
    ax[1].imshow(np.sqrt(im2.squeeze()), cmap='gray')


def rpc_affine_approximation(rpc, p):
    """
    Compute the first order Taylor approximation of an RPC projection function.

    Args:
        rpc: instance of the rpc_model.RPCModel class
        p: lon, lat, h coordinates

    Return:
        array of shape (3, 4) representing the affine camera matrix equal to the
        first order Taylor approximation of the RPC projection function at point p.
    """
    p = ad.adnumber(p)
    q = rpc.projection(*p)
    J = ad.jacobian(q, p)

    A = np.zeros((3, 4))
    A[:2, :3] = J
    A[:2, 3] = np.array(q) - np.dot(J, p)
    A[2, 3] = 1
    return A


def affine_fundamental_matrix(p, q):
    """
    Compute the affine fundamental matrix from two affine camera matrices.

    Args:
        p, q: arrays of shape (3, 4) representing the input camera matrices.

    Return:
        array of shape (3, 3) representing the affine fundamental matrix computed
        with the formula 17.3 (p. 412) from Hartley & Zisserman book (2nd ed.).
    """
    X0 = p[[1, 2], :]
    X1 = p[[2, 0], :]
    X2 = p[[0, 1], :]
    Y0 = q[[1, 2], :]
    Y1 = q[[2, 0], :]
    Y2 = q[[0, 1], :]

    F = np.zeros((3, 3))
    F[0, 2] = np.linalg.det(np.vstack([X2, Y0]))
    F[1, 2] = np.linalg.det(np.vstack([X2, Y1]))
    F[2, 0] = np.linalg.det(np.vstack([X0, Y2]))
    F[2, 1] = np.linalg.det(np.vstack([X1, Y2]))
    F[2, 2] = np.linalg.det(np.vstack([X2, Y2]))

    return F


def rectifying_similarities_from_affine_fundamental_matrix(F, debug=False):
    """
    Computes two similarities from an affine fundamental matrix.

    Args:
        F: 3x3 numpy array representing the input fundamental matrix
        debug (optional, default is False): boolean flag to activate verbose
            mode

    Returns:
        S, S': two similarities such that, when used to resample the two images
            related by the fundamental matrix, the resampled images are
            stereo-rectified.
    """
    # check that the input matrix is an affine fundamental matrix
    assert(np.shape(F) == (3, 3))
    assert(np.linalg.matrix_rank(F) == 2)
    np.testing.assert_allclose(F[:2, :2], np.zeros((2, 2)))

    # notations
    a = F[2, 0]
    b = F[2, 1]
    c = F[0, 2]
    d = F[1, 2]
    e = F[2, 2]

    # rotations
    r = np.sqrt(a*a + b*b)
    s = np.sqrt(c*c + d*d)
    R1 = (1.0 / r) * np.array([[b, -a], [a, b]])
    R2 = (1.0 / s) * np.array([[-d, c], [-c, -d]])

    # zoom and translation
    z = np.sqrt(r / s)
    t = 0.5 * e / np.sqrt(r * s)

    if debug:
        theta_1 = get_angle_from_cos_and_sin(b / r, a / r)
        print("reference image:")
        print("\trotation: %f deg" % np.rad2deg(theta_1))
        print("\tzoom: %f" % z)
        print("\tvertical translation: %f" % t)
        print()
        theta_2 = get_angle_from_cos_and_sin(-d / s, -c / s)
        print("secondary image:")
        print("\trotation: %f deg" % np.rad2deg(theta_2))
        print("\tzoom: %f" % (1.0 / z))
        print("\tvertical translation: %f" % -t)

    # output similarities
    S1 = np.zeros((3, 3))
    S1[0:2, 0:2] = z * R1
    S1[1, 2] = t
    S1[2, 2] = 1

    S2 = np.zeros((3, 3))
    S2[0:2, 0:2] = (1.0 / z) * R2
    S2[1, 2] = -t
    S2[2, 2] = 1

    return S1, S2


def affine_transformation(x, xx):
    """
    Estimate a 2D affine transformation from a list of point matches.

    Args:
        x:  Nx2 numpy array, containing a list of points
        xx: Nx2 numpy array, containing the list of corresponding points

    Returns:
        3x3 numpy array, representing in homogeneous coordinates an affine
        transformation that maps the points of x onto the points of xx.

    This function implements the Gold-Standard algorithm for estimating an
    affine homography, described in Hartley & Zisserman page 130 (second
    edition).
    """
    # check that there are at least 3 points
    if len(x) < 3:
        print("ERROR: affine_transformation needs at least 3 matches")
        return np.eye(3)

    # translate the input points so that the centroid is at the origin.
    t = -np.mean(x,  0)
    tt = -np.mean(xx, 0)
    x = x + t
    xx = xx + tt

    # compute the Nx4 matrix A
    A = np.hstack((x, xx))

    # two singular vectors corresponding to the two largest singular values of
    # matrix A. See Hartley and Zissermann for details.  These are the first
    # two lines of matrix V (because np.linalg.svd returns V^T)
    U, S, V = np.linalg.svd(A)
    v1 = V[0, :]
    v2 = V[1, :]

    # compute blocks B and C, then H
    tmp = np.vstack((v1, v2)).T
    assert(np.shape(tmp) == (4, 2))
    B = tmp[0:2, :]
    C = tmp[2:4, :]
    H = np.dot(C, np.linalg.inv(B))

    # return A
    A = np.eye(3)
    A[0:2, 0:2] = H
    A[0:2, 2] = np.dot(H, t) - tt
    return A


def rectifying_affine_transforms(rpc1, rpc2, aoi, z=0):
    """
    Compute two affine transforms that rectify two images over a given AOI.

    Args:
        rpc1, rpc2 (rpc_model.RPCModel): two RPC camera models
        aoi (geojson.Polygon): area of interest

    Return:
        S1, S2 (2D arrays): two numpy arrays of shapes (3, 3) representing the
            rectifying affine transforms in homogeneous coordinates
        w, h (ints): minimal width and height of the rectified image crops
            needed to cover the AOI
        P1, P2 (2D arrays): two numpy arrays of shapes (3, 3) representing the
            affine camera matrices used to approximate the rpc camera models
    """
    # center of the AOI
    lons, lats = np.asarray(aoi['coordinates'][0][:4]).T
    lon, lat = np.mean([lons, lats], axis=1)

    # affine projection matrices that approximate the rpc models around the
    # center of the AOI
    P1 = rpc_affine_approximation(rpc1, (lon, lat, z))
    P2 = rpc_affine_approximation(rpc2, (lon, lat, z))

    # affine fundamental matrix associated to our two images
    F = affine_fundamental_matrix(P1, P2)

    # compute rectifying similarities
    S1, S2 = rectifying_similarities_from_affine_fundamental_matrix(F)

    # affine correction of S2 to register the ground (horizontal plane at z)
    q1 = S1 @ P1 @ [lons, lats, [z, z, z, z], [1, 1, 1, 1]]
    q2 = S2 @ P2 @ [lons, lats, [z, z, z, z], [1, 1, 1, 1]]
    S2 = affine_transformation(q2[:2].T, q1[:2].T) @ S2

    # shift the rectified images so that their top-left corners fall on (0, 0)
    x1, y1, w1, h1 = utils.bounding_box_of_projected_aoi(rpc1, aoi, z=z,
                                                         homography=S1)
    x2, y2, w2, h2 = utils.bounding_box_of_projected_aoi(rpc2, aoi, z=z,
                                                         homography=S2)
    S1 = matrix_translation(-x1, -0.5 * (y1 + y2)) @ S1
    S2 = matrix_translation(-x2, -0.5 * (y1 + y2)) @ S2

    w = int(round(max(w1, w2)))
    h = int(round(max(h1, h2)))
    return S1, S2, w, h, P1, P2


def match_pair(a, b):
    """
    Find SIFT matching points in two images represented as numpy arrays.

    Args:
        a, b (arrays): two numpy arrays containing the input images to match

    Return:
        pts1, pts2: two lists of pairs of coordinates of matching points
    """
    a = utils.simple_equalization_8bit(a)
    b = utils.simple_equalization_8bit(b)

    # KP
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(a, None)
    kp2, des2 = sift.detectAndCompute(b, None)
#    kp = sift.detect(a, None)
#    img = cv2.drawKeypoints(a, kp, b)
#    display_image(img)
#    cv2.imwrite('sift_keypoints.jpg', img)

    # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)


    # cv2.drawMatchesKnn expects list of lists as matches.
#    img3 = cv2.drawMatchesKnn(a,kp1,b,kp2,good,a,flags=2)

    pts1 = np.asarray(pts1)
    pts2 = np.asarray(pts2)
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]

#   display_image(img3)
    return  pts1, pts2


def sift_roi(file1, file2, aoi, z):
    """
    Args:
        file1, file2: filename of two satellite images
        aoi: area of interest
        z: base height for the aoi

    Returns:
        q1, q2: numpy arrays with the coordinates of the matching points in the
            original (full-size) image domains
    """
    # image crops
    crop1, x1, y1 = utils.crop_aoi(file1, aoi, z=z)
    crop2, x2, y2 = utils.crop_aoi(file2, aoi, z=z)

    # sift keypoint matches
    p1, p2 = match_pair(crop1, crop2)
    q1 = utils.points_apply_homography(matrix_translation(x1, y1), p1)
    q2 = utils.points_apply_homography(matrix_translation(x2, y2), p2)
    return q1, q2


def affine_crop(input_path, A, w, h):
    """
    Apply an affine transform to an image.

    Args:
        input_path (string): path or url to the input image
        A (numpy array): 3x3 array representing an affine transform in
            homogeneous coordinates
        w, h (ints): width and height of the output image

    Return:
        numpy array of shape (h, w) containing a subset of the transformed
        image. The subset is the rectangle between points 0, 0 and w, h.
    """
    # determine the rectangle that we need to read in the input image
    output_rectangle = [[0, 0], [w, 0], [w, h], [0, h]]
    x, y, w0, h0 = utils.bounding_box2D(utils.points_apply_homography(np.linalg.inv(A),
                                                                      output_rectangle))
    x, y = np.floor((x, y)).astype(int)
    w0, h0 = np.ceil((w0, h0)).astype(int)

    # crop the needed rectangle in the input image
    with utils.rio_open(input_path, 'r') as src:
        aoi = src.read(indexes=1, window=((y, y + h0), (x, x + w0)))

    # compensate the affine transform for the crop
    B = A @ matrix_translation(x, y)

    # apply the affine transform
    out = ndimage.affine_transform(aoi.T, np.linalg.inv(B), output_shape=(w, h)).T
    return out


def rectify_aoi(file1, file2, aoi, z=None):
    """
    Args:
        file1, file2: filename of two satellite images
        aoi: area of interest
        z (float, optional): base altitude with respect to WGS84 ellipsoid. If
            None, z is retrieved from srtm.

    Returns:
        rect1, rect2: numpy arrays with the images
        S1, S2: transformation matrices from the coordinate system of the original images
        disp_min, disp_max: horizontal disparity range
        P1, P2: affine rpc approximations of the two images computed during the rectification
    """
    # read the RPC coefficients
    rpc1 = utils.rpc_from_geotiff(file1)
    rpc2 = utils.rpc_from_geotiff(file2)

    # get the altitude of the center of the AOI
    if z is None:
        lon, lat = np.mean(aoi['coordinates'][0][:4], axis=0)
        z = srtm4.srtm4(lon, lat)

    # compute rectifying affine transforms
    S1, S2, w, h, P1, P2 = rectifying_affine_transforms(rpc1, rpc2, aoi, z=z)

    # compute sift keypoint matches
    q1, q2 = sift_roi(file1, file2, aoi, z)

    # transform the matches to the domain of the rectified images
    q1 = utils.points_apply_homography(S1, q1)
    q2 = utils.points_apply_homography(S2, q2)

    # pointing correction (y_shift)
    y_shift = np.median(q2 - q1, axis=0)[1]
    S2 = matrix_translation(0, -y_shift) @ S2

    # rectify the crops
    rect1 = affine_crop(file1, S1, w, h)
    rect2 = affine_crop(file2, S2, w, h)

    # disparity range bounds
    kpts_disps = (q2 - q1)[:, 0]
    disp_min = np.percentile(kpts_disps, 2)
    disp_max = np.percentile(kpts_disps, 100 - 2)

    return rect1, rect2, S1, S2, disp_min, disp_max, P1, P2

def affine_crop_truth(xyList, A, w, h):
    """
    Apply an affine transform to an image.

    Args:
        input_path (string): path or url to the input image
        A (numpy array): 3x3 array representing an affine transform in
            homogeneous coordinates
        w, h (ints): width and height of the output image

    Return:
        numpy array of shape (h, w) containing a subset of the transformed
        image. The subset is the rectangle between points 0, 0 and w, h.
    """
    # determine the rectangle that we need to read in the input image
    output_rectangle = [[0, 0], [w, 0], [w, h], [0, h]]
    x, y, w0, h0 = utils.bounding_box2D(utils.points_apply_homography(np.linalg.inv(A),
                                                                      output_rectangle))
    x, y = np.floor((x, y)).astype(int)

    # compensate the affine transform for the crop
    B = A @ matrix_translation(x, y)

    # apply the affine transform
    out = ndimage.affine_transform(xyList, np.linalg.inv(B), output_shape=(w, h)).T
    return out


def rectify_aoi_truth(file1, file2, aoi, xyLeft, xyRight, z=None):
    """
    Args:
        file1, file2: filename of two satellite images
        aoi: area of interest
        z (float, optional): base altitude with respect to WGS84 ellipsoid. If
            None, z is retrieved from srtm.

    Returns:
        rect1, rect2: numpy arrays with the images
        S1, S2: transformation matrices from the coordinate system of the original images
        disp_min, disp_max: horizontal disparity range
        P1, P2: affine rpc approximations of the two images computed during the rectification
    """
    # read the RPC coefficients
    rpc1 = utils.rpc_from_geotiff(file1)
    rpc2 = utils.rpc_from_geotiff(file2)

    # get the altitude of the center of the AOI
    if z is None:
        lon, lat = np.mean(aoi['coordinates'][0][:4], axis=0)
        z = srtm4.srtm4(lon, lat)

    # compute rectifying affine transforms
    S1, S2, w, h, P1, P2 = rectifying_affine_transforms(rpc1, rpc2, aoi, z=z)

    # compute sift keypoint matches
    q1, q2 = sift_roi(file1, file2, aoi, z)

    # transform the matches to the domain of the rectified images
    q1 = utils.points_apply_homography(S1, q1)
    q2 = utils.points_apply_homography(S2, q2)

    # pointing correction (y_shift)
    y_shift = np.median(q2 - q1, axis=0)[1]
    S2 = matrix_translation(0, -y_shift) @ S2

    # rectify the crops
    rect1 = affine_crop(file1, S1, w, h)
    rect2 = affine_crop(file2, S2, w, h)

    xyRect1 = affine_crop_truth(xyLeft, S1, w, h)
    xyRect2 = affine_crop_truth(xyRight, S2, w, h)


    # disparity range bounds
    kpts_disps = (q2 - q1)[:, 0]
    disp_min = np.percentile(kpts_disps, 2)
    disp_max = np.percentile(kpts_disps, 100 - 2)

    return rect1, rect2, xyRect1,xyRect2
