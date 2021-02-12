"""
* affine triangulation
* display point clouds
* DEM projection

Copyright (C) 2018, Enric Meinhardt-Llopis <enric.meinhardt@cmla.ens-cachan.fr>
Copyright (C) 2018, Carlo de Franchis <carlo.de-franchis@ens-cachan.fr>
Copyright (C) 2018, Gabriele Facciolo <facciolo@cmla.ens-cachan.fr>
"""

import os
import ctypes
from ctypes import c_int, c_float, c_double, byref, POINTER
from numpy.ctypeslib import ndpointer
import numpy as np
from scipy import ndimage

import libs2p.common
from libs2p.config import cfg
import sys

here = os.path.dirname(os.path.abspath(__file__))
#lib_path = os.path.join(os.path.dirname(here), 'lib', 'disp_to_h.so')
version = sys.version_info
lib_path = os.path.join(sys.prefix, 'lib/python%d.%d/site-packages/'%(int(version[0]),int(version[1])), 'lib', 'disp_to_h.so')
lib = ctypes.CDLL(lib_path)


class RPCStruct(ctypes.Structure):
    """
    ctypes version of the RPC C struct defined in rpc.h.
    """
    _fields_ = [("numx", c_double * 20),
                ("denx", c_double * 20),
                ("numy", c_double * 20),
                ("deny", c_double * 20),
                ("scale", c_double * 3),
                ("offset", c_double * 3),
                ("inumx", c_double * 20),
                ("idenx", c_double * 20),
                ("inumy", c_double * 20),
                ("ideny", c_double * 20),
                ("iscale", c_double * 3),
                ("ioffset", c_double * 3),
                ("dmval", c_double * 4),
                ("imval", c_double * 4)]

    def __init__(self, rpc):
        """
        Args:
            rpc (rpcm.RPCModel): rpc model
        """
        self.offset[0] = rpc.col_offset
        self.offset[1] = rpc.row_offset
        self.offset[2] = rpc.alt_offset
        self.ioffset[0] = rpc.lon_offset
        self.ioffset[1] = rpc.lat_offset
        self.ioffset[2] = rpc.alt_offset

        self.scale[0] = rpc.col_scale
        self.scale[1] = rpc.row_scale
        self.scale[2] = rpc.alt_scale
        self.iscale[0] = rpc.lon_scale
        self.iscale[1] = rpc.lat_scale
        self.iscale[2] = rpc.alt_scale

        for i in range(20):
            self.inumx[i] = rpc.col_num[i]
            self.idenx[i] = rpc.col_den[i]
            self.inumy[i] = rpc.row_num[i]
            self.ideny[i] = rpc.row_den[i]

        if hasattr(rpc, 'lat_num'):
            for i in range(20):
                self.numx[i] = rpc.lon_num[i]
                self.denx[i] = rpc.lon_den[i]
                self.numy[i] = rpc.lat_num[i]
                self.deny[i] = rpc.lat_den[i]
        else:
            for i in range(20):
                self.numx[i] = np.nan
                self.denx[i] = np.nan
                self.numy[i] = np.nan
                self.deny[i] = np.nan



def disp_to_xyz(rpc1, rpc2, H1, H2, disp, mask, utm_zone, img_bbx=None, A=None):
    """
    Compute a height map from a disparity map, using RPC camera models.

    Args:
        rpc1, rpc2 (rpcm.RPCModel): camera models
        H1, H2 (arrays): 3x3 numpy arrays defining the rectifying homographies
        disp, mask (array): 2D arrays of shape (h, w) representing the diparity
            and mask maps
        utm_zone (int): desired UTM zone number (between 1 and 60) for the
            output xyz map
        img_bbx (4-tuple): col_min, col_max, row_min, row_max defining the
            unrectified image domain to process.
        A (array): 3x3 array with the pointing correction matrix for im2

    Returns:
        xyz: array of shape (h, w, 3) where each pixel contains the UTM
            easting, northing, and altitude of the triangulated point.
        err: array of shape (h, w) where each pixel contains the triangulation
            error
    """
    # copy rpc coefficients to an RPCStruct object
    rpc1_c_struct = RPCStruct(rpc1)
    rpc2_c_struct = RPCStruct(rpc2)

    # handle optional arguments
    if A is not None:  # apply pointing correction
        H2 = np.dot(H2, np.linalg.inv(A))

    if img_bbx is None:
        img_bbx = (-np.inf, np.inf, -np.inf, np.inf)

    # define the argument types of the disp_to_xyz function from disp_to_h.so
    h, w = disp.shape
    lib.disp_to_xyz.argtypes = (ndpointer(dtype=c_float, shape=(h, w, 3)),
                                ndpointer(dtype=c_float, shape=(h, w)),
                                ndpointer(dtype=c_float, shape=(h, w)),
                                ndpointer(dtype=c_float, shape=(h, w)),
                                ndpointer(dtype=c_float, shape=(h, w)),
                                c_int, c_int,
                                ndpointer(dtype=c_double, shape=(9,)),
                                ndpointer(dtype=c_double, shape=(9,)),
                                POINTER(RPCStruct), POINTER(RPCStruct), c_int,
                                ndpointer(dtype=c_float, shape=(4,)))


    # call the disp_to_xyz function from disp_to_h.so
    xyz =  np.zeros((h, w, 3), dtype='float32')
    err =  np.zeros((h, w), dtype='float32')
    dispx = disp.astype('float32')
    dispy = np.zeros((h, w), dtype='float32')
    msk = mask.astype('float32')
    lib.disp_to_xyz(xyz, err, dispx, dispy, msk, w, h,
                    H1.flatten(), H2.flatten(),
                    byref(rpc1_c_struct), byref(rpc2_c_struct), utm_zone,
                    np.asarray(img_bbx, dtype='float32'))

    return xyz, err


def triangulation_affine(PA, PB, x1, y1, x2, y2):
    """
    Triangulate a (list of) match(es) between two images of affine cameras.

    Arguments:
        PA, PB : affine (projection) camera matrices of the two images
        x1, y1 : pixel coordinates in the domain of the first image
        x2, y2 : pixel coordinates in the domain of the second image

    Return value: a 4-tuple (lon, lat, h, e)
        lon, lat, h, e : coordinates of the 3D point(s), reprojection error
    """

    # build projection and localization matrices as 4x4 homogeneous maps
    # (x,y,h,1) <-> (lon,lat,h,1)
    PA = np.vstack([ PA[0:2], [0,0,1,0], [0,0,0,1]])  # pick only first two rows
    PB = np.vstack([ PB[0:2], [0,0,1,0], [0,0,0,1]])  # pick only first two rows
    LA = np.linalg.inv(PA)  # inverse of a 4x4 matrix

    # affine epipolar correspondence
    E = PB @ LA

    # Now, the linear system E * (x1, y1, h, 1) = (x2, y2, h, 1)
    # has two different equations and one unknown h.  We solve it by
    # least squares (a simple projection, in this case).

    # give names to the 8 non-trivial coefficients of E
    a, b, p, r = E[0]
    c, d, q, s = E[1]

    # coefficients of the affine triangulation
    f = [-p*a - q*c, -p*b - q*d, p, q, -p*r - q*s ] / (p*p + q*q)

    # apply the triangulation (first use of the input points)
    h = f[0]*x1 + f[1]*y1 + f[2]*x2 + f[3]*y2 + f[4]

    # finish the computation and return the 4 required numbers (or vectors)
    lon = LA[0,0] * x1 + LA[0,1] * y1 + LA[0,2] * h + LA[0,3]
    lat = LA[1,0] * x1 + LA[1,1] * y1 + LA[1,2] * h + LA[1,3]
    ex = E[0,0] * x1 + E[0,1] * y1 + E[0,2] * h + E[0,3] - x2
    ey = E[1,0] * x1 + E[1,1] * y1 + E[1,2] * h + E[1,3] - y2
    e = ex * ex + ey * ey
    return lon, lat, h, e


def rastio_z(row,col,xyz,x,y):
    """
    mchen 12,March,2020

    :param dmap: a disparity map between two rectified images
    :param xyz:  a matrix of size Nx3 (where N is the number of finite disparites
              in dmap) this matrix contains the coordinates of the 3d points in
              "lon,lat,h" or "e,n,h"
    :return:
    """

    xyh = np.full((row,col),0,dtype=int)
    scale_x = x.shape[0]
    for k in range(scale_x):
        i = y[k]
        j = x[k]
        xyh[i][j]=np.round(xyz[k][2])

    return xyh


def triangulate_disparities(dmap, rpc1, rpc2, S1, S2, PA, PB,):
    """
    Triangulate a disparity map

    Arguments:
        dmap : a disparity map between two rectified images
        rpc1, rpc2 : calibration data of each image
        S1, S2 : rectifying affine maps (from the domain of the fullsize images)
        PA, PB : the affine approximations of rpc1 and rpc2 (not always used)

    Return:
        xyz : a matrix of size Nx3 (where N is the number of finite disparites
              in dmap) this matrix contains the coordinates of the 3d points in
              "lon,lat,h" or "e,n,h"
    """
    from libs2p.utils import utm_from_lonlat

    # 1. unroll all the valid (finite) disparities of dmap into a vector
    m = np.isfinite(dmap.flatten())
    x = np.argwhere(np.isfinite(dmap))[:,1]  # attention to order of the indices
    y = np.argwhere(np.isfinite(dmap))[:,0]
    d = dmap.flatten()[m]

    # 2. for each disparity

    # 2.1. produce a pair in the original domain by composing with S1 and S2
    p = np.linalg.inv(S1) @ np.vstack( (x+0, y, np.ones(len(d))) )
    q = np.linalg.inv(S2) @ np.vstack( (x+d, y, np.ones(len(d))) )

    # 2.2. triangulate the pair of image points to find a 3D point (in UTM)
    lon, lat, h, e = triangulation_affine(PA,PB, p[0,:],p[1,:], q[0,:],q[1,:])
    #east, north, _, _ = np.vectorize(utm.from_latlon)(lat, lon)
    east, north = utm_from_lonlat(lon, lat)

    # 2.3. add this point to the output list
    xyz = np.vstack((east, north, h)).T

    # mChen 2020.3.12 just for display
    xyh = rastio_z(dmap.shape[0],dmap.shape[1],xyz,x,y)

    return xyz, xyh


# code for projection into UTM coordinates
from numba import jit

# utility function for computing the average at each cell
@jit
def reduceavg(w, h,  ix, iy, z):
    D_sum = np.zeros((h,w))  #-np.ones((h,w))*np.inf
    D_cnt = np.zeros((h,w))
    for t in range(len(ix)):
        if iy[t]<0 or ix[t]<0 or iy[t]>=h or ix[t]>=w:
            continue
        ty = iy[t]
        tx = ix[t]
        D_sum[ty,tx] += z[t]
        D_cnt[ty,tx] += 1

    D_sum /= D_cnt*(D_cnt>0) + 1*(D_cnt==0) # needed for computing average

    return D_sum


# projects them into a grid defined by the bunding box (emin, emax, nmin, nmax)
def project_cloud_into_utm_grid(xyz, emin, emax, nmin, nmax, resolution=1):
    """
    Project a point cloud into an utm grid to produce a DEM
    The algorithm averages all the points that fall into each square of the grid

    Arguments:
        xyz : a Nx3 matrix representing a point cloud in (lon,lat,h) coordinates
        emin,emax,nmin,nmax : a bounding box in UTM coordinates
        resolution : the target resolution in meters (by default, 1 meter)

    Return:
        dem : a 2D array of heights in meters
    """

    # width and height of the image domain
    w = int(np.ceil((emax - emin)/resolution))
    h = int(np.ceil((nmax - nmin)/resolution))

    # extract and quantize columns
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]

    ix = np.asarray((x - emin)/resolution, dtype="int")
    iy = np.asarray((nmax - y)/resolution, dtype="int")

    # per-cell average
    dem = reduceavg (w,h,  ix, iy, z )

    return dem

