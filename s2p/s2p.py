#!/usr/bin/env python

# s2p - Satellite Stereo Pipeline
# Copyright (C) 2015, Carlo de Franchis <carlo.de-franchis@polytechnique.org>
# Copyright (C) 2015, Gabriele Facciolo <facciolo@cmla.ens-cachan.fr>
# Copyright (C) 2015, Enric Meinhardt <enric.meinhardt@cmla.ens-cachan.fr>
# Copyright (C) 2015, Julien Michel <julien.michel@cnes.fr>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import sys
import os.path
import json
import datetime
import argparse
import numpy as np
import subprocess
import multiprocessing
import collections
import shutil
import rasterio
from skimage.transform import warp, AffineTransform
import matplotlib.pyplot as plt
from skimage import io as io

from   libs2p.config import cfg
import libs2p.common
import libs2p.parallel
import libs2p.initialization
import libs2p.pointing_accuracy
import libs2p.rectification
import libs2p.block_matching
import libs2p.masking
import libs2p.triangulation
import libs2p.fusion
import libs2p.rasterization
import libs2p.visualisation
import libs2p.ply
import libs2p.matchfstBi

import LAF.computeLafDisp
import LibMccnn

def pointing_correction(tile, i):
    """
    Compute the translation that corrects the pointing error on a pair of tiles.

    Args:
        tile: dictionary containing the information needed to process the tile
        i: index of the processed pair
    """
    x, y, w, h = tile['coordinates']
    out_dir = os.path.join(tile['dir'], 'pair_{}'.format(i))
    img1 = cfg['images'][0]['img']
    rpc1 = cfg['images'][0]['rpcm']
    img2 = cfg['images'][i]['img']
    rpc2 = cfg['images'][i]['rpcm']

    # correct pointing error
    print('correcting pointing on tile {} {} pair {}...'.format(x, y, i))
    try:
        A, m = libs2p.pointing_accuracy.compute_correction(img1, img2, rpc1, rpc2, x, y, w, h)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        stderr = os.path.join(out_dir, 'stderr.log')
        with open(stderr, 'w') as f:
            f.write('ERROR during pointing correction with cmd: %s\n' % e[0]['command'])
            f.write('Stop processing this pair\n')
        return

    if A is not None:  # A is the correction matrix
        np.savetxt(os.path.join(out_dir, 'pointing.txt'), A, fmt='%6.3f')
    if m is not None:  # m is the list of sift matches
        np.savetxt(os.path.join(out_dir, 'sift_matches.txt'), m, fmt='%9.3f')
        np.savetxt(os.path.join(out_dir, 'center_keypts_sec.txt'),
                   np.mean(m[:, 2:], 0), fmt='%9.3f')
        if cfg['debug']:
            libs2p.visualisation.plot_matches(img1, img2, rpc1, rpc2, m, x, y, w, h,
                                       os.path.join(out_dir,
                                                    'sift_matches_pointing.png'))


def global_pointing_correction(tiles):
    """
    Compute the global pointing corrections for each pair of images.

    Args:
        tiles: list of tile dictionaries
    """
    for i in range(1, len(cfg['images'])):
        out = os.path.join(cfg['out_dir'], 'global_pointing_pair_%d.txt' % i)
        l = [os.path.join(t['dir'], 'pair_%d' % i) for t in tiles]
        np.savetxt(out, libs2p.pointing_accuracy.global_from_local(l),
                   fmt='%12.6f')
        if cfg['clean_intermediate']:
            for d in l:
                libs2p.common.remove(os.path.join(d, 'center_keypts_sec.txt'))


def rectification_pair(tile, i):
    """
    Rectify a pair of images on a given tile.

    Args:
        tile: dictionary containing the information needed to process a tile.
        i: index of the processed pair
    """
    print('Line 1 of rectification_pair function')
    out_dir = os.path.join(tile['dir'], 'pair_{}'.format(i))

    x, y, w, h = tile['coordinates']
    img1 = cfg['images'][0]['img']
    rpc1 = cfg['images'][0]['rpcm']
    img2 = cfg['images'][i]['img']
    rpc2 = cfg['images'][i]['rpcm']
    pointing = os.path.join(cfg['out_dir'],
                            'global_pointing_pair_{}.txt'.format(i))

    outputs = ['disp_min_max.txt', 'rectified_ref.tif', 'rectified_sec.tif']

    if os.path.exists(os.path.join(out_dir, 'stderr.log')):
        print('rectification: stderr.log exists')
        print('pair_{} not processed on tile {} {}'.format(i, x, y))
        return

    print('rectifying tile {} {} pair {}...'.format(x, y, i))
    try:
        A = np.loadtxt(os.path.join(out_dir, 'pointing.txt'))
    except IOError:
        A = np.loadtxt(pointing)
    try:
        m = np.loadtxt(os.path.join(out_dir, 'sift_matches.txt'))
    except IOError:
        m = None

    x, y, w, h = tile['coordinates']

    cur_dir = os.path.join(tile['dir'],'pair_{}'.format(i))
    for n in tile['neighborhood_dirs']:
        nei_dir = os.path.join(tile['dir'], n, 'pair_{}'.format(i))
        if os.path.exists(nei_dir) and not os.path.samefile(cur_dir, nei_dir):
            sift_from_neighborhood = os.path.join(nei_dir, 'sift_matches.txt')
            try:
                m_n = np.loadtxt(sift_from_neighborhood)
                # added sifts in the ellipse of semi axes : (3*w/4, 3*h/4)
                m_n = m_n[np.where(np.linalg.norm([(m_n[:,0]-(x+w/2))/w,
                                                   (m_n[:,1]-(y+h/2))/h],
                                                  axis=0) < 3.0/4)]
                if m is None:
                    m = m_n
                else:
                    m = np.concatenate((m, m_n))
            except IOError:
                print('%s does not exist' % sift_from_neighborhood)

    rect1 = os.path.join(out_dir, 'rectified_ref.tif')
    rect2 = os.path.join(out_dir, 'rectified_sec.tif')
    H1, H2, disp_min, disp_max = libs2p.rectification.rectify_pair(img1, img2,
                                                            rpc1, rpc2,
                                                            x, y, w, h,
                                                            rect1, rect2, A, m,
                                                            method=cfg['rectification_method'],
                                                            hmargin=cfg['horizontal_margin'],
                                                            vmargin=cfg['vertical_margin'])
    np.savetxt(os.path.join(out_dir, 'H_ref.txt'), H1, fmt='%12.6f')
    np.savetxt(os.path.join(out_dir, 'H_sec.txt'), H2, fmt='%12.6f')
    np.savetxt(os.path.join(out_dir, 'disp_min_max.txt'), [disp_min, disp_max],
                            fmt='%3.1f')

    if cfg['clean_intermediate']:
        libs2p.common.remove(os.path.join(out_dir,'pointing.txt'))
        libs2p.common.remove(os.path.join(out_dir,'sift_matches.txt'))


def stereo_matching_mccnn_basic(tile,i):
    """
    Compute the disparity of a pair of images on a given tile.

    Args:
        tile: dictionary containing the information needed to process a tile.
        i: index of the processed pair
    """
    out_dir = os.path.join(tile['dir'], 'pair_{}'.format(i))
    x, y = tile['coordinates'][:2]

    outputs = ['rectified_mask.png', 'rectified_disp.tif']

    if os.path.exists(os.path.join(out_dir, 'stderr.log')):
        print('disparity estimation: stderr.log exists')
        print('pair_{} not processed on tile {} {}'.format(i, x, y))
        return

    print('estimating disparity on tile {} {} pair {}...'.format(x, y, i))
    print(os.path.join(out_dir, 'disp_min_max.txt'))
    rect1 = os.path.join(out_dir, 'rectified_ref.tif')
    rect2 = os.path.join(out_dir, 'rectified_sec.tif')
    disp = os.path.join(out_dir, 'rectified_disp.tif')
    mask = os.path.join(out_dir, 'rectified_mask.png')
    disp_min, disp_max = np.loadtxt(os.path.join(out_dir, 'disp_min_max.txt'))
    disp_min, disp_max = int(np.floor(disp_min)), int(np.ceil(disp_max)) #

    # Determine the model to be loaded
    resume = os.path.join(cfg['mccnn_model_dir'], 'checkpoint')
    #sub_pixel_interpolation = cfg['sub_pixel_interpolation']
    #filter_cost_volumes = cfg['filter_cost_volumes']
    
    libs2p.matchfstBi.compute_disparity_mccnn_basic(rect1, rect2, disp, disp_min, disp_max,resume)
    #libs2p.matchfstBi.compute_disparity_mccnn_basic(rect2, rect1, disp, disp_min, disp_max,resume)
    libs2p.block_matching.create_rejection_mask(disp, rect1, rect2, mask)
    
    # add margin around masked pixels
    libs2p.masking.erosion(mask, mask, cfg['msk_erosion'])

    if cfg['clean_intermediate']:
        if len(cfg['images']) > 2:
            libs2pcommon.remove(rect1)
        libs2p.common.remove(rect2)
        libs2p.common.remove(os.path.join(out_dir,'disp_min_max.txt'))

def stereo_matching_mccnn_laf(tile,i):
    """
    Compute the disparity of a pair of images on a given tile.

    Args:
        tile: dictionary containing the information needed to process a tile.
        i: index of the processed pair
    """
    out_dir = os.path.join(tile['dir'], 'pair_{}'.format(i))
    x, y = tile['coordinates'][:2]

    outputs = ['rectified_mask.png', 'rectified_disp.tif']

    if os.path.exists(os.path.join(out_dir, 'stderr.log')):
        print('disparity estimation: stderr.log exists')
        print('pair_{} not processed on tile {} {}'.format(i, x, y))
        return

    print('estimating disparity on tile {} {} pair {}...'.format(x, y, i))
    print(os.path.join(out_dir, 'disp_min_max.txt'))
    rect1 = os.path.join(out_dir, 'rectified_ref.tif')
    rect2 = os.path.join(out_dir, 'rectified_sec.tif')
    disp = os.path.join(out_dir, 'rectified_disp.tif')
    mask = os.path.join(out_dir, 'rectified_mask.png')
    disp_min, disp_max = np.loadtxt(os.path.join(out_dir, 'disp_min_max.txt'))
    disp_min, disp_max = int(np.floor(disp_min)), int(np.ceil(disp_max)) #

    # Determine the model to be loaded
    resume = os.path.join(cfg['mccnn_model_dir'], 'checkpoint')
    #sub_pixel_interpolation = cfg['sub_pixel_interpolation']
    #filter_cost_volumes = cfg['filter_cost_volumes']
    
    #libs2p.matchfstBi.compute_disparity_mccnn_basic(rect1, rect2, disp, disp_min, disp_max,resume)
    libs2p.matchfstBi.compute_disparity_mccnn_laf(rect1, rect2, disp, disp_min, disp_max,resume)
    #libs2p.matchfstBi.compute_disparity_mccnn_basic(rect2, rect1, disp, disp_min, disp_max,resume)
    libs2p.block_matching.create_rejection_mask(disp, rect1, rect2, mask)
    
    # add margin around masked pixels
    libs2p.masking.erosion(mask, mask, cfg['msk_erosion'])

    # computing LAF-Net confidence and disparity
    LAF_model_dir = cfg["laf_model_dir"]
    img_path = out_dir
    confidence_disparity,confidence_map = LAF.computeLafDisp.compute_confidence_disparity(LAF_model_dir, img_path)

    # save as pgm and pfm
    disp = os.path.join(out_dir, 'selected_disp.tif')
    confidence_disparity = -confidence_disparity
    io.imsave(disp, confidence_disparity)
    conf = os.path.join(out_dir, 'selected_conf.tif')
    confidence_map = (confidence_map*(256.*256.-1)).astype(np.uint16)
    io.imsave(conf, confidence_map)

    if cfg['clean_intermediate']:
        if len(cfg['images']) > 2:
            libs2pcommon.remove(rect1)
        libs2p.common.remove(rect2)
        libs2p.common.remove(os.path.join(out_dir,'disp_min_max.txt'))


def stereo_matching_sgm(tile,i):
    """
    Compute the disparity of a pair of images on a given tile.

    Args:
        tile: dictionary containing the information needed to process a tile.
        i: index of the processed pair
    """
    out_dir = os.path.join(tile['dir'], 'pair_{}'.format(i))
    x, y = tile['coordinates'][:2]

    outputs = ['rectified_mask.png', 'rectified_disp.tif']

    if os.path.exists(os.path.join(out_dir, 'stderr.log')):
        print('disparity estimation: stderr.log exists')
        print('pair_{} not processed on tile {} {}'.format(i, x, y))
        return

    print('estimating disparity on tile {} {} pair {}...'.format(x, y, i))
    print(os.path.join(out_dir, 'disp_min_max.txt'))
    rect1 = os.path.join(out_dir, 'rectified_ref.tif')
    rect2 = os.path.join(out_dir, 'rectified_sec.tif')
    disp = os.path.join(out_dir, 'rectified_disp.tif')
    mask = os.path.join(out_dir, 'rectified_mask.png')
    disp_min, disp_max = np.loadtxt(os.path.join(out_dir, 'disp_min_max.txt'))
    disp_min, disp_max = int(np.floor(disp_min)), int(np.ceil(disp_max)) #
    print((disp_min, disp_max))
    #libs2p.block_matching.compute_disparity_map(rect1, rect2, disp, mask,
    #                                     cfg['matching_algorithm'], disp_min,
    #                                     disp_max)
    libs2p.block_matching.compute_disparity_map(rect1, rect2, disp, mask,
                                         cfg['matching_algorithm'], disp_min,
                                         disp_max)
    
    #libs2p.matchfstBi.compute_disparity_mccnn(rect1, rect2, disp, disp_min, disp_max)
    libs2p.block_matching.create_rejection_mask(disp, rect1, rect2, mask)
    
    # add margin around masked pixels
    libs2p.masking.erosion(mask, mask, cfg['msk_erosion'])

    if cfg['clean_intermediate']:
        if len(cfg['images']) > 2:
            libs2pcommon.remove(rect1)
        libs2p.common.remove(rect2)
        libs2p.common.remove(os.path.join(out_dir,'disp_min_max.txt'))



def disparity_to_height(tile, i):
    """
    Compute a height map from the disparity map of a pair of image tiles.

    Args:
        tile: dictionary containing the information needed to process a tile.
        i: index of the processed pair.
    """
    out_dir = os.path.join(tile['dir'], 'pair_{}'.format(i))
    x, y, w, h = tile['coordinates']

    if os.path.exists(os.path.join(out_dir, 'stderr.log')):
        print('triangulation: stderr.log exists')
        print('pair_{} not processed on tile {} {}'.format(i, x, y))
        return

    print('triangulating tile {} {} pair {}...'.format(x, y, i))
    rpc1 = cfg['images'][0]['rpcm']
    rpc2 = cfg['images'][i]['rpcm']
    H_ref = np.loadtxt(os.path.join(out_dir, 'H_ref.txt'))
    H_sec = np.loadtxt(os.path.join(out_dir, 'H_sec.txt'))
    disp = os.path.join(out_dir, 'rectified_disp.tif')
    mask = os.path.join(out_dir, 'rectified_mask.png')
    pointing = os.path.join(cfg['out_dir'],
                            'global_pointing_pair_{}.txt'.format(i))

    with rasterio.open(disp, 'r') as f:
        disp_img = f.read().squeeze()
    with rasterio.open(mask, 'r') as f:
        mask_rect_img = f.read().squeeze()
    height_map = triangulation.height_map(x, y, w, h, rpc1, rpc2, H_ref, H_sec,
                                          disp_img, mask_rect_img,
                                          int(cfg['utm_zone'][:-1]),
                                          A=np.loadtxt(pointing))

    # write height map to a file
    common.rasterio_write(os.path.join(out_dir, 'height_map.tif'), height_map)

    if cfg['clean_intermediate']:
        common.remove(H_ref)
        common.remove(H_sec)
        common.remove(disp)
        common.remove(mask)


def disparity_to_ply(tile):
    """
    Compute a point cloud from the disparity map of a pair of image tiles.

    Args:
        tile: dictionary containing the information needed to process a tile.
    """
    out_dir = os.path.join(tile['dir'])
    ply_file = os.path.join(out_dir, 'cloud.ply')
    plyextrema = os.path.join(out_dir, 'plyextrema.txt')
    x, y, w, h = tile['coordinates']
    rpc1 = cfg['images'][0]['rpcm']
    rpc2 = cfg['images'][1]['rpcm']

    if os.path.exists(os.path.join(out_dir, 'stderr.log')):
        print('triangulation: stderr.log exists')
        print('pair_1 not processed on tile {} {}'.format(x, y))
        return

    print('triangulating tile {} {}...'.format(x, y))
    # This function is only called when there is a single pair (pair_1)
    H_ref = os.path.join(out_dir, 'pair_1', 'H_ref.txt')
    H_sec = os.path.join(out_dir, 'pair_1', 'H_sec.txt')
    pointing = os.path.join(cfg['out_dir'], 'global_pointing_pair_1.txt')
    disp  = os.path.join(out_dir, 'pair_1', 'rectified_disp.tif')
    extra = os.path.join(out_dir, 'pair_1', 'rectified_disp_confidence.tif')
    if not os.path.exists(extra):
        extra = ''
    mask_rect = os.path.join(out_dir, 'pair_1', 'rectified_mask.png')
    mask_orig = os.path.join(out_dir, 'mask.png')

    # prepare the image needed to colorize point cloud
    colors = os.path.join(out_dir, 'rectified_ref.png')
    if cfg['images'][0]['clr']:
        hom = np.loadtxt(H_ref)
        # We want rectified_ref.png and rectified_ref.tif to have the same size
        with rasterio.open(os.path.join(out_dir, 'pair_1', 'rectified_ref.tif')) as f:
            ww, hh = f.width, f.height
        libs2p.common.image_apply_homography(colors, cfg['images'][0]['clr'], hom, ww, hh)
    else:
        libs2p.common.image_qauto(os.path.join(out_dir, 'pair_1', 'rectified_ref.tif'), colors)

    # compute the point cloud
    with rasterio.open(disp, 'r') as f:
        disp_img = f.read().squeeze()
    with rasterio.open(mask_rect, 'r') as f:
        mask_rect_img = f.read().squeeze()
    xyz_array, err = libs2p.triangulation.disp_to_xyz(rpc1, rpc2,
                                               np.loadtxt(H_ref), np.loadtxt(H_sec),
                                               disp_img, mask_rect_img,
                                               int(cfg['utm_zone'][:-1]),
                                               img_bbx=(x, x+w, y, y+h),
                                               A=np.loadtxt(pointing))

    # 3D filtering
    if cfg['3d_filtering_r'] and cfg['3d_filtering_n']:
        triangulation.filter_xyz(xyz_array, cfg['3d_filtering_r'],
                                 cfg['3d_filtering_n'], cfg['gsd'])

    # flatten the xyz array into a list and remove nan points
    xyz_list = xyz_array.reshape(-1, 3)
    valid = np.all(np.isfinite(xyz_list), axis=1)

    # write the point cloud to a ply file
    with rasterio.open(colors, 'r') as f:
        img = f.read()
    colors_list = img.transpose(1, 2, 0).reshape(-1, img.shape[0])
    libs2p.ply.write_3d_point_cloud_to_ply(ply_file, xyz_list[valid],
                                    colors=colors_list[valid],
                                    extra_properties=None,
                                    extra_properties_names=None,
                                    comments=["created by S2P",
                                              "projection: UTM {}".format(cfg['utm_zone'])])

    # compute the point cloud extrema (xmin, xmax, xmin, ymax)
    libs2p.common.run("plyextrema %s %s" % (ply_file, plyextrema))

    if cfg['clean_intermediate']:
        libs2p.common.remove(H_ref)
        libs2p.common.remove(H_sec)
        libs2p.common.remove(disp)
        libs2p.common.remove(mask_rect)
        libs2p.common.remove(mask_orig)
        libs2p.common.remove(colors)
        libs2p.common.remove(os.path.join(out_dir, 'pair_1', 'rectified_ref.tif'))


def mean_heights(tile):
    """
    """
    w, h = tile['coordinates'][2:]
    n = len(cfg['images']) - 1
    maps = np.empty((h, w, n))
    for i in range(n):
        try:
            with rasterio.open(os.path.join(tile['dir'], 'pair_{}'.format(i + 1),
                                            'height_map.tif'), 'r') as f:
                maps[:, :, i] = f.read(1)
        except RuntimeError:  # the file is not there
            maps[:, :, i] *= np.nan

    validity_mask = maps.sum(axis=2)  # sum to propagate nan values
    validity_mask += 1 - validity_mask  # 1 on valid pixels, and nan on invalid

    # save the n mean height values to a txt file in the tile directory
    np.savetxt(os.path.join(tile['dir'], 'local_mean_heights.txt'),
               [np.nanmean(validity_mask * maps[:, :, i]) for i in range(n)])


def global_mean_heights(tiles):
    """
    """
    local_mean_heights = [np.loadtxt(os.path.join(t['dir'], 'local_mean_heights.txt'))
                          for t in tiles]
    global_mean_heights = np.nanmean(local_mean_heights, axis=0)
    for i in range(len(cfg['images']) - 1):
        np.savetxt(os.path.join(cfg['out_dir'],
                                'global_mean_height_pair_{}.txt'.format(i+1)),
                   [global_mean_heights[i]])


def heights_fusion(tile):
    """
    Merge the height maps computed for each image pair and generate a ply cloud.

    Args:
        tile: a dictionary that provides all you need to process a tile
    """
    tile_dir = tile['dir']
    height_maps = [os.path.join(tile_dir, 'pair_%d' % (i + 1), 'height_map.tif')
                   for i in range(len(cfg['images']) - 1)]

    # remove spurious matches
    if cfg['cargarse_basura']:
        for img in height_maps:
            common.cargarse_basura(img, img)

    # load global mean heights
    global_mean_heights = []
    for i in range(len(cfg['images']) - 1):
        x = np.loadtxt(os.path.join(cfg['out_dir'],
                                    'global_mean_height_pair_{}.txt'.format(i+1)))
        global_mean_heights.append(x)

    # merge the height maps (applying mean offset to register)
    fusion.merge_n(os.path.join(tile_dir, 'height_map.tif'), height_maps,
                   global_mean_heights, averaging=cfg['fusion_operator'],
                   threshold=cfg['fusion_thresh'])

    if cfg['clean_intermediate']:
        for f in height_maps:
            common.remove(f)


def heights_to_ply(tile):
    """
    Generate a ply cloud.

    Args:
        tile: a dictionary that provides all you need to process a tile
    """
    # merge the n-1 height maps of the tile (n = nb of images)
    heights_fusion(tile)

    # compute a ply from the merged height map
    out_dir = tile['dir']
    x, y, w, h = tile['coordinates']
    plyfile = os.path.join(out_dir, 'cloud.ply')
    plyextrema = os.path.join(out_dir, 'plyextrema.txt')
    height_map = os.path.join(out_dir, 'height_map.tif')

    # H is the homography transforming the coordinates system of the original
    # full size image into the coordinates system of the crop
    H = np.dot(np.diag([1, 1, 1]), common.matrix_translation(-x, -y))
    colors = os.path.join(out_dir, 'ref.tif')
    if cfg['images'][0]['clr']:
        common.image_crop_gdal(cfg['images'][0]['clr'], x, y, w, h, colors)
    else:
        common.image_qauto(common.image_crop_gdal(cfg['images'][0]['img'], x, y,
                                                 w, h), colors)

    triangulation.height_map_to_point_cloud(plyfile, height_map,
                                            cfg['images'][0]['rpcm'], H, colors,
                                            utm_zone=cfg['utm_zone'])

    # compute the point cloud extrema (xmin, xmax, xmin, ymax)
    common.run("plyextrema %s %s" % (plyfile, plyextrema))

    if cfg['clean_intermediate']:
        common.remove(height_map)
        common.remove(colors)
        common.remove(os.path.join(out_dir, 'mask.png'))

def plys_to_dsm(tile):
    """
    Generates DSM from plyfiles (cloud.ply)

    Args:
        tile: a dictionary that provides all you need to process a tile
    """
    out_dsm  = os.path.join(tile['dir'], 'dsm.tif')
    out_conf = os.path.join(tile['dir'], 'confidence.tif')

    res = cfg['dsm_resolution']
    if 'utm_bbx' in cfg:
        bbx = cfg['utm_bbx']
        global_xoff = bbx[0]
        global_yoff = bbx[3]
    else:
        global_xoff = 0  # arbitrary reference
        global_yoff = 0

    xmin, xmax, ymin, ymax = np.loadtxt(os.path.join(tile['dir'], "plyextrema.txt"))

    if not all(np.isfinite([xmin, xmax, ymin, ymax])):  # then the ply is empty
        return

    # compute xoff, yoff, xsize, ysize considering final dsm
    xoff = global_xoff + np.floor((xmin - global_xoff) / res) * res
    xsize = int(1 + np.floor((xmax - xoff) / res))

    yoff = global_yoff + np.ceil((ymax - global_yoff) / res) * res
    ysize = int(1 - np.floor((ymin - yoff) / res))

    roi = xoff, yoff, xsize, ysize

    clouds = [os.path.join(tile['dir'], n_dir, 'cloud.ply') for n_dir in tile['neighborhood_dirs']]
    raster, profile = libs2p.rasterization.plyflatten_from_plyfiles_list(clouds,
                                                                  resolution=res,
                                                                  roi=roi,
                                                                  radius=cfg['dsm_radius'],
                                                                  sigma=cfg['dsm_sigma'])

    # save output image with utm georeferencing
    libs2p.common.rasterio_write(out_dsm, raster[:, :, 0], profile=profile)

    # export confidence (optional)
    if raster.shape[-1] == 5:
        libs2p.common.rasterio_write(out_conf, raster[:, :, 4], profile=profile)


def global_xyz(tiles):
    """
    Generates XYZ from plyfiles (cloud.ply)
    # mChen 20200304

    Args:
        tiles: a dictionary that provides all tile
    Output File:
        ply.xyz: the UTM cloud point for visualizer.jar(metric)
    """
    out_xyz = os.path.join(cfg['out_dir'], 'ply.xyz')

    clouds_list = [os.path.join(t['dir'], 'cloud.ply') for t in tiles]
    full_cloud = list()
    for cloud in clouds_list:
        cloud_data, _ = libs2p.ply.read_3d_point_cloud_from_ply(cloud)
        full_cloud.append(cloud_data.astype(np.float64))

    full_cloud = np.concatenate(full_cloud)

    # The copy() method will reorder to C-contiguous order by default:
    full_cloud = full_cloud.copy()
    xyz = full_cloud[:, 0:3]

    # save output data
    np.savetxt(out_xyz, xyz, fmt="%.18f")

def global_dsm(tiles):
    """
    """
    out_dsm_vrt = os.path.join(cfg['out_dir'], 'dsm.vrt')
    out_dsm_tif = os.path.join(cfg['out_dir'], 'dsm.tif')

    dsms_list = [os.path.join(t['dir'], 'dsm.tif') for t in tiles]
    dsms = '\n'.join(d for d in dsms_list if os.path.exists(d))

    input_file_list = os.path.join(cfg['out_dir'], 'gdalbuildvrt_input_file_list.txt')

    with open(input_file_list, 'w') as f:
        f.write(dsms)

    libs2p.common.run("gdalbuildvrt -vrtnodata nan -input_file_list %s %s" % (input_file_list,
                                                                       out_dsm_vrt))

    res = cfg['dsm_resolution']

    if 'utm_bbx' in cfg:
        bbx = cfg['utm_bbx']
        xoff = bbx[0]
        yoff = bbx[3]
        xsize = int(np.ceil((bbx[1]-bbx[0]) / res))
        ysize = int(np.ceil((bbx[3]-bbx[2]) / res))
        projwin = "-projwin %s %s %s %s" % (xoff, yoff,
                                            xoff + xsize * res,
                                            yoff - ysize * res)
    else:
        projwin = ""

    libs2p.common.run(" ".join(["gdal_translate",
                         "-co TILED=YES -co BIGTIFF=IF_SAFER",
                         "%s %s %s" % (projwin, out_dsm_vrt, out_dsm_tif)]))

    # EXPORT CONFIDENCE
    out_conf_vrt = os.path.join(cfg['out_dir'], 'confidence.vrt')
    out_conf_tif = os.path.join(cfg['out_dir'], 'confidence.tif')

    dsms_list = [os.path.join(t['dir'], 'confidence.tif') for t in tiles]
    dems_list_ok = [d for d in dsms_list if os.path.exists(d)]
    dsms = '\n'.join(dems_list_ok)

    input_file_list = os.path.join(cfg['out_dir'], 'gdalbuildvrt_input_file_list2.txt')

    if len(dems_list_ok) > 0:

        with open(input_file_list, 'w') as f:
            f.write(dsms)

        common.run("gdalbuildvrt -vrtnodata nan -input_file_list %s %s" % (input_file_list,
                                                                           out_conf_vrt))

        common.run(" ".join(["gdal_translate",
                             "-co TILED=YES -co BIGTIFF=IF_SAFER",
                             "%s %s %s" % (projwin, out_conf_vrt, out_conf_tif)]))
def compute_disparity_map(tiles):
    # Derive the output directory
    for tile in tiles:
        out_dir = os.path.join(tile['dir'])
        # Derive the coordinates (x,y,w,h) of the tile being processed		
        x, y, w, h = tile['coordinates']
        # Get the rpc values from the cofiguraton file
        rpc1 = cfg['images'][0]['rpcm']
        rpc2 = cfg['images'][1]['rpcm']
        # Load the homographies
        H_ref = os.path.join(out_dir, 'pair_1', 'H_ref.txt')
        H_sec = os.path.join(out_dir, 'pair_1', 'H_sec.txt')
        # Get the path of the disparity map
        disp  = os.path.join(out_dir, 'pair_1', 'rectified_disp.tif')
        # Get the rectified mask
        mask_rect = os.path.join(out_dir, 'pair_1', 'rectified_mask.png')
        # Get the original mask
        mask_orig = os.path.join(out_dir, 'mask.png')
        # Load the rectified reference image
        colors = os.path.join(out_dir, 'rectified_ref.png')
        # Uniform requantization between min and max intensity
        libs2p.common.image_qauto(os.path.join(out_dir, 'pair_1', 'rectified_ref.tif'), colors)
        # Derive the filename of a file containing the global pointing pair_1 data
        pointing = os.path.join(cfg['out_dir'], 'global_pointing_pair_1.txt')
        # Load the homographies
        H1 = np.loadtxt(H_ref)
        H2 = np.loadtxt(H_sec)
        # Derive the image bounding box
        img_bbx = (x, x+w, y, y+h)
        # Load the global pointing data 
        A = np.loadtxt(pointing)

        # compute the point cloud
        with rasterio.open(disp, 'r') as f:
            disp_img = f.read().squeeze()
        with rasterio.open(colors, 'r') as f:
            img_color = f.read().squeeze()
        # copy rpc coefficients to an RPCStruct object
        rpc1_c_struct = libs2p.triangulation.RPCStruct(rpc1)
        rpc2_c_struct = libs2p.triangulation.RPCStruct(rpc2)
        
        # handle optional arguments
        if A is not None:  # apply pointing correction
            H2 = np.dot(H2, np.linalg.inv(A))
        
        # Derive the affine transform
        tform = AffineTransform(H1)
        
        # Compute the inverse affine transformation
        image = warp(img_color/255, tform.inverse, output_shape=(2364, 2452))
        
        plt.figure
        plt.imshow(image,cmap='gray')#,cmap='jet', interpolation='None',vmin=-24,vmax=25)
        plt.colorbar()

        plt.show()

        
       
     		
    '''    
    # compute the point cloud
    xyz_array, err = libs2p.triangulation.disp_to_xyz(rpc1, rpc2,
                                               np.loadtxt(H_ref), np.loadtxt(H_sec),
                                               disp_img, mask_rect_img,
                                               int(cfg['utm_zone'][:-1]),
                                               img_bbx=(x, x+w, y, y+h),
                                               A=np.loadtxt(pointing))

    # 3D filtering
    if cfg['3d_filtering_r'] and cfg['3d_filtering_n']:
        triangulation.filter_xyz(xyz_array, cfg['3d_filtering_r'],
                                 cfg['3d_filtering_n'], cfg['gsd'])

    # flatten the xyz array into a list and remove nan points
    xyz_list = xyz_array.reshape(-1, 3)
    valid = np.all(np.isfinite(xyz_list), axis=1)

    # write the point cloud to a ply file
    with rasterio.open(colors, 'r') as f:
        img = f.read()
    colors_list = img.transpose(1, 2, 0).reshape(-1, img.shape[0])
    libs2p.ply.write_3d_point_cloud_to_ply(ply_file, xyz_list[valid],
                                    colors=colors_list[valid],
                                    extra_properties=None,
                                    extra_properties_names=None,
                                    comments=["created by S2P",
                                              "projection: UTM {}".format(cfg['utm_zone'])])

    # compute the point cloud extrema (xmin, xmax, xmin, ymax)
    libs2p.common.run("plyextrema %s %s" % (ply_file, plyextrema))

    if cfg['clean_intermediate']:
        libs2p.common.remove(H_ref)
        libs2p.common.remove(H_sec)
        libs2p.common.remove(disp)
        libs2p.common.remove(mask_rect)
        libs2p.common.remove(mask_orig)
        libs2p.common.remove(colors)
        libs2p.common.remove(os.path.join(out_dir, 'pair_1', 'rectified_ref.tif'))
    '''

def main(user_cfg):
    """
    Launch the s2p pipeline with the parameters given in a json file.

    Args:
        user_cfg: user config dictionary
    """
    libs2p.common.print_elapsed_time.t0 = datetime.datetime.now()
    libs2p.initialization.build_cfg(user_cfg)
    libs2p.initialization.make_dirs()
    # multiprocessing setup
    nb_workers = multiprocessing.cpu_count()  # nb of available cores
    if cfg['max_processes'] is not None:
        nb_workers = cfg['max_processes']

    tw, th = libs2p.initialization.adjust_tile_size()
    tiles_txt = os.path.join(cfg['out_dir'], 'tiles.txt')
    tiles = libs2p.initialization.tiles_full_info(tw, th, tiles_txt, create_masks=True)
    if not tiles:
        print('ERROR: the ROI is not seen in two images or is totally masked.')
        return

    # initialisation: write the list of tilewise json files to outdir/tiles.txt
    with open(tiles_txt, 'w') as f:
        for t in tiles:
            print(t['json'], file=f)

    n = len(cfg['images'])
    tiles_pairs = [(t, i) for i in range(1, n) for t in tiles]
    
    # local-pointing step:
    print('correcting pointing locally...')
    libs2p.parallel.launch_calls(pointing_correction, tiles_pairs, nb_workers)
 
    # global-pointing step:
    print('correcting pointing globally...')
    global_pointing_correction(tiles)
    libs2p.common.print_elapsed_time()

    # rectification step:
    print('rectifying tiles...')
    
    libs2p.parallel.launch_calls(rectification_pair, tiles_pairs, nb_workers)

    print('stereo vision...')
    if cfg['matching_algorithm'] == 'mccnn_basic':
        stereo_matching = stereo_matching_mccnn_basic
    elif cfg['matching_algorithm'] == 'sgbm':
        stereo_matching = stereo_matching_sgm
    elif cfg['matching_algorithm'] == 'mccnn_laf':
        stereo_matching = stereo_matching_mccnn_laf
        
    libs2p.parallel.launch_calls(stereo_matching, tiles_pairs, nb_workers)
    #for tile in tiles:
    #    stereo_matching(tile,1)

    # Note: The output of the estimated disparities also includes several nan values. At 
    # some point these will be converted to -9999 which is marked in the analysis to indicated 
    # an unknown pixel value.
    
	# Derive height map directly	
    # triangulation step:
    print('triangulating tiles...')
    libs2p.parallel.launch_calls(disparity_to_ply, tiles, nb_workers)

    # mChen 20200304
    print('computing global point clouds...')
    global_xyz(tiles)
    libs2p.common.print_elapsed_time()

    # local-dsm-rasterization step:
    print('computing DSM by tile...')
    libs2p.parallel.launch_calls(plys_to_dsm, tiles, nb_workers)

    # global-dsm-rasterization step:
    print('computing global DSM...')
    global_dsm(tiles)
    libs2p.common.print_elapsed_time()

    # cleanup
    libs2p.common.garbage_cleanup()
    libs2p.common.print_elapsed_time(since_first_call=True)
    

def make_path_relative_to_file(path, f):
    return os.path.join(os.path.abspath(os.path.dirname(f)), path)


def read_tiles(tiles_file):
    tiles = []
    outdir = os.path.dirname(tiles_file)

    with open(tiles_file) as f:
        tiles = f.readlines()

    # Strip trailing \n
    tiles = list(map(str.strip,tiles))
    tiles = [os.path.join(outdir, t) for t in tiles]

    return tiles


def read_config_file(config_file):
    """
    Read a json configuration file and interpret relative paths.

    If any input or output path is a relative path, it is interpreted as
    relative to the config_file location (and not relative to the current
    working directory). Absolute paths are left unchanged.
    """
    with open(config_file, 'r') as f:
        user_cfg = json.load(f)

    # output paths
    if not os.path.isabs(user_cfg['out_dir']):
        print('WARNING: out_dir is a relative path. It is interpreted with '
              'respect to {} location (not cwd)'.format(config_file))
        user_cfg['out_dir'] = make_path_relative_to_file(user_cfg['out_dir'],
                                                         config_file)
        print('out_dir is: {}'.format(user_cfg['out_dir']))

    # ROI paths
    for k in ["roi_kml", "roi_geojson"]:
        if k in user_cfg and isinstance(user_cfg[k], str) and not os.path.isabs(user_cfg[k]):
            user_cfg[k] = make_path_relative_to_file(user_cfg[k], config_file)

    # input paths
    for img in user_cfg['images']:
        for d in ['img', 'rpc', 'clr', 'cld', 'roi', 'wat']:
            if d in img and isinstance(img[d], str) and not os.path.isabs(img[d]):
                img[d] = make_path_relative_to_file(img[d], config_file)

    return user_cfg
