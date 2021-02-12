# Copyright (C) 2019, David Youssefi (CNES) <david.youssefi@cnes.fr>


import re
import os
import numpy as np
import ctypes
from numpy.ctypeslib import ndpointer

import affine
import sys
import libs2p.geographiclib, libs2p.ply

# TODO: This is kind of ugly. Cleaner way to do this is to update
# LD_LIBRARY_PATH, which we should do once we have a proper config file
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#lib = ctypes.CDLL(os.path.join(parent_dir, 'lib', 'libplyflatten.so'))
version = sys.version_info
lib = ctypes.CDLL(os.path.join(sys.prefix, 'lib/python%d.%d/site-packages/'%(int(version[0]),int(version[1])), 'lib', 'libplyflatten.so'))


class InvalidPlyCommentsError(Exception):
    pass


def plyflatten(cloud,
               xoff, yoff,
               resolution,
               xsize, ysize,
               radius, sigma):
    """
    Projects a points cloud into the raster band(s) of a raster image

    Args:
        cloud: A nb_points x (2+nb_extra_columns) numpy array:
            | x0 y0 [z0 r0 g0 b0 ...] |
            | x1 y1 [z1 r1 g1 b1 ...] |
            | ...                     |
            | xN yN [zN rN gN bN ...] |
            x, y give positions of the points into the final raster, the "extra
            columns" give the values
        xoff, yoff: offset position (upper left corner) considering the georeferenced image
        resolution: resolution of the output georeferenced image
        xsize, ysize: size of the georeferenced image
        radius: controls the spread of the blob from each point
        sigma: radius of influence for each point (unit: pixel)

    Returns;
        A numpy array of shape (ysize, xsize, nb_extra_columns)
    """
    nb_points, nb_extra_columns = cloud.shape[0], cloud.shape[1] - 2
    raster_shape = (xsize * ysize, nb_extra_columns)

    # Set expected args and return types
    lib.rasterize_cloud.argtypes = (ndpointer(dtype=ctypes.c_double,
                                              shape=np.shape(cloud)),
                                    ndpointer(dtype=ctypes.c_float,
                                              shape=raster_shape),
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_double, ctypes.c_double,
                                    ctypes.c_double,
                                    ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_float)

    # Call rasterize_cloud function from libplyflatten.so
    raster = np.zeros(raster_shape, dtype='float32')
    lib.rasterize_cloud(np.ascontiguousarray(cloud.astype(np.float64)),
                        raster,
                        nb_points,
                        nb_extra_columns,
                        xoff, yoff,
                        resolution,
                        xsize, ysize,
                        radius, sigma)

    # Transform result into a numpy array
    raster = raster.reshape((ysize, xsize, nb_extra_columns))

    return raster


def plyflatten_from_plyfiles_list(clouds_list, resolution, radius=0, roi=None, sigma=None):
    """
    Projects a points cloud into the raster band(s) of a raster image (points clouds as files)

    Args:
        clouds_list: list of cloud.ply files
        resolution: resolution of the georeferenced output raster file
        roi: region of interest: (xoff, yoff, xsize, ysize), compute plyextrema if None

    Returns:
        raster: georeferenced raster
        profile: profile for rasterio
    """
    # read points clouds
    full_cloud = list()
    for cloud in clouds_list:
        cloud_data, _ = libs2p.ply.read_3d_point_cloud_from_ply(cloud)
        full_cloud.append(cloud_data.astype(np.float64))

    full_cloud = np.concatenate(full_cloud)

    # region of interest (compute plyextrema if roi is None)
    if roi is not None:
        xoff, yoff, xsize, ysize = roi
    else:
        xx = full_cloud[:, 0]
        yy = full_cloud[:, 1]
        xmin = np.amin(xx)
        xmax = np.amax(xx)
        ymin = np.amin(yy)
        ymax = np.amax(yy)

        xsize = int(1 + np.floor((xmax - xmin) / resolution))
        ysize = int(1 + np.floor((ymax - ymin) / resolution))
        xoff = (xmax + xmin - resolution * xsize) / 2
        yoff = (ymax + ymin + resolution * ysize) / 2

    # The copy() method will reorder to C-contiguous order by default:
    full_cloud = full_cloud.copy()
    sigma = float("inf") if sigma is None else sigma
    raster = plyflatten(full_cloud, xoff, yoff, resolution,
                        xsize, ysize,
                        radius, sigma)

    utm_zone = utm_zone_from_ply(clouds_list[0])
    utm_proj = libs2p.geographiclib.utm_proj(utm_zone)

    # construct profile dict
    profile = dict()
    profile['tiled'] = True
    profile['nodata'] = float('nan')
    profile['crs'] = utm_proj.srs
    profile['transform'] = affine.Affine(resolution, 0.0, xoff,
                                         0.0, -resolution, yoff)

    return raster, profile


def utm_zone_from_ply(ply_path):
    _, comments = libs2p.ply.read_3d_point_cloud_from_ply(ply_path)
    regex = r"^projection: UTM (\d{1,2}[NS])"
    utm_zone = None
    for comment in comments:
        s = re.search(regex, comment)
        if s:
            utm_zone = s.group(1)

    if not utm_zone:
        raise InvalidPlyCommentsError(
            "Invalid header comments {} for ply file {}".format(comments, ply_path)
        )

    return utm_zone
