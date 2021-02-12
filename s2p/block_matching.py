# Copyright (C) 2015, Carlo de Franchis <carlo.de-franchis@ens-cachan.fr>
# Copyright (C) 2015, Gabriele Facciolo <facciolo@cmla.ens-cachan.fr>
# Copyright (C) 2015, Enric Meinhardt <enric.meinhardt@cmla.ens-cachan.fr>
# Copyright (C) 2015, Julien Michel <julien.michel@cnes.fr>

import os
import numpy as np

import libs2p.common
from libs2p.config import cfg
import s2p

def rectify_secondary_tile_only(algo):
    if algo in ['tvl1_2d']:
        return True
    else:
        return False


def create_rejection_mask(disp, im1, im2, mask):
    """
    Create rejection mask (0 means rejected, 1 means accepted)
    Keep only the points that are matched and present in both input images

    Args:
        disp: path to the input disparity map
        im1, im2: rectified stereo pair
        mask: path to the output rejection mask
    """
    tmp1 = libs2p.common.tmpfile('.tif')
    tmp2 = libs2p.common.tmpfile('.tif')
    libs2p.common.run(["plambda", disp, "x 0 join", "-o", tmp1])
    libs2p.common.run(["backflow", tmp1, im2, tmp2])
    libs2p.common.run(["plambda", disp, im1, tmp2, "x isfinite y isfinite z isfinite and and", "-o", mask])


def compute_disparity_map(im1, im2, disp, mask, algo, disp_min=None,
                          disp_max=None, timeout=cfg['mgm_timeout'], extra_params=''):
    """
    Runs a block-matching binary on a pair of stereo-rectified images.

    Args:
        im1, im2: rectified stereo pair
        disp: path to the output diparity map
        mask: path to the output rejection mask
        algo: string used to indicate the desired binary. Currently it can be
            one among 'hirschmuller02', 'hirschmuller08',
            'hirschmuller08_laplacian', 'hirschmuller08_cauchy', 'sgbm',
            'msmw', 'tvl1', 'mgm', 'mgm_multi' and 'micmac'
        disp_min : smallest disparity to consider
        disp_max : biggest disparity to consider
        timeout: time in seconds after which the disparity command will
            raise an error if it hasn't returned.
            Only applies to `mgm*` algorithms.
        extra_params: optional string with algorithm-dependent parameters
    """
    if rectify_secondary_tile_only(algo) is False:
        disp_min = [disp_min]
        disp_max = [disp_max]

    # limit disparity bounds
    np.alltrue(len(disp_min) == len(disp_max))
    for dim in range(len(disp_min)):
        if disp_min[dim] is not None and disp_max[dim] is not None:
            image_size = libs2p.common.image_size_gdal(im1)
            if disp_max[dim] - disp_min[dim] > image_size[dim]:
                center = 0.5 * (disp_min[dim] + disp_max[dim])
                disp_min[dim] = int(center - 0.5 * image_size[dim])
                disp_max[dim] = int(center + 0.5 * image_size[dim])

        # round disparity bounds
        if disp_min[dim] is not None:
            disp_min[dim] = int(np.floor(disp_min[dim]))
        if disp_max is not None:
            disp_max[dim] = int(np.ceil(disp_max[dim]))

    if rectify_secondary_tile_only(algo) is False:
        disp_min = disp_min[0]
        disp_max = disp_max[0]

    # define environment variables
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(cfg['omp_num_threads'])

    # call the block_matching binary
    if algo == 'hirschmuller02':
        bm_binary = 'subpix.sh'
        common.run('{0} {1} {2} {3} {4} {5} {6} {7}'.format(bm_binary, im1, im2, disp, mask, disp_min,
                                                            disp_max, extra_params))
        # extra_params: LoG(0) regionRadius(3)
        #    LoG: Laplacian of Gaussian preprocess 1:enabled 0:disabled
        #    regionRadius: radius of the window

    if algo == 'hirschmuller08':
        bm_binary = 'callSGBM.sh'
        common.run('{0} {1} {2} {3} {4} {5} {6} {7}'.format(bm_binary, im1, im2, disp, mask, disp_min,
                                                            disp_max, extra_params))
        # extra_params: regionRadius(3) P1(default) P2(default) LRdiff(1)
        #    regionRadius: radius of the window
        #    P1, P2 : regularization parameters
        #    LRdiff: maximum difference between left and right disparity maps

    if algo == 'hirschmuller08_laplacian':
        bm_binary = 'callSGBM_lap.sh'
        common.run('{0} {1} {2} {3} {4} {5} {6} {7}'.format(bm_binary, im1, im2, disp, mask, disp_min,
                                                            disp_max, extra_params))
    if algo == 'hirschmuller08_cauchy':
        bm_binary = 'callSGBM_cauchy.sh'
        common.run('{0} {1} {2} {3} {4} {5} {6} {7}'.format(bm_binary, im1, im2, disp, mask, disp_min,
                                                            disp_max, extra_params))
    if algo == 'sgbm':
        # opencv sgbm function implements a modified version of Hirschmuller's
        # Semi-Global Matching (SGM) algorithm described in "Stereo Processing
        # by Semiglobal Matching and Mutual Information", PAMI, 2008

        p1 = 8  # penalizes disparity changes of 1 between neighbor pixels
        p2 = 32  # penalizes disparity changes of more than 1
        # it is required that p2 > p1. The larger p1, p2, the smoother the disparity

        win = 3  # matched block size. It must be a positive odd number
        lr = 1  # maximum difference allowed in the left-right disparity check
        cost = libs2p.common.tmpfile('.tif')
        libs2p.common.run('sgbm {} {} {} {} {} {} {} {} {} {}'.format(im1, im2,
                                                               disp, cost,
                                                               disp_min,
                                                               disp_max,
                                                               win, p1, p2, lr))

        create_rejection_mask(disp, im1, im2, mask)

    if algo == 'tvl1':
        tvl1 = 'callTVL1.sh'
        common.run('{0} {1} {2} {3} {4}'.format(tvl1, im1, im2, disp, mask),
                   env)

    if algo == 'tvl1_2d':
        tvl1 = 'callTVL1.sh'
        common.run('{0} {1} {2} {3} {4} {5}'.format(tvl1, im1, im2, disp, mask,
                                                    1), env)


    if algo == 'msmw':
        bm_binary = 'iip_stereo_correlation_multi_win2'
        common.run('{0} -i 1 -n 4 -p 4 -W 5 -x 9 -y 9 -r 1 -d 1 -t -1 -s 0 -b 0 -o 0.25 -f 0 -P 32 -m {1} -M {2} {3} {4} {5} {6}'.format(bm_binary, disp_min, disp_max, im1, im2, disp, mask))

    if algo == 'msmw2':
        bm_binary = 'iip_stereo_correlation_multi_win2_newversion'
        common.run('{0} -i 1 -n 4 -p 4 -W 5 -x 9 -y 9 -r 1 -d 1 -t -1 -s 0 -b 0 -o -0.25 -f 0 -P 32 -D 0 -O 25 -c 0 -m {1} -M {2} {3} {4} {5} {6}'.format(
                bm_binary, disp_min, disp_max, im1, im2, disp, mask), env)

    if algo == 'msmw3':
        bm_binary = 'msmw'
        common.run('{0} -m {1} -M {2} -il {3} -ir {4} -dl {5} -kl {6}'.format(
                bm_binary, disp_min, disp_max, im1, im2, disp, mask))

    if algo == 'mgm':
        env['MEDIAN'] = '1'
        env['CENSUS_NCC_WIN'] = str(cfg['census_ncc_win'])
        env['TSGM'] = '3'

        nb_dir = cfg['mgm_nb_directions']

        conf = '{}_confidence.tif'.format(os.path.splitext(disp)[0])

        libs2p.common.run(
            '{executable} '
            '-r {disp_min} -R {disp_max} '
            '-s vfit '
            '-t census '
            '-O {nb_dir} '
            '-confidence_consensusL {conf} '
            '{im1} {im2} {disp}'.format(
                executable='mgm',
                disp_min=disp_min,
                disp_max=disp_max,
                nb_dir=nb_dir,
                conf=conf,
                im1=im1,
                im2=im2,
                disp=disp,
            ),
            env=env,
            timeout=timeout,
        )

        create_rejection_mask(disp, im1, im2, mask)


    if algo == 'mgm_multi_lsd':


        ref = im1
        sec = im2


        wref = common.tmpfile('.tif')
        wsec = common.tmpfile('.tif')
        # TODO TUNE LSD PARAMETERS TO HANDLE DIRECTLY 12 bits images?
        # image dependent weights based on lsd segments
        image_size = common.image_size_gdal(ref)
        #TODO refactor this command to not use shell=True
        common.run('qauto %s | \
                   lsd  -  - | \
                   cut -d\' \' -f1,2,3,4   | \
                   pview segments %d %d | \
                   plambda -  "255 x - 255 / 2 pow 0.1 fmax" -o %s'%(ref,image_size[0], image_size[1],wref),
                   shell=True)
        # image dependent weights based on lsd segments
        image_size = common.image_size_gdal(sec)
        #TODO refactor this command to not use shell=True
        common.run('qauto %s | \
                   lsd  -  - | \
                   cut -d\' \' -f1,2,3,4   | \
                   pview segments %d %d | \
                   plambda -  "255 x - 255 / 2 pow 0.1 fmax" -o %s'%(sec,image_size[0], image_size[1],wsec),
                   shell=True)


        env['REMOVESMALLCC'] = str(cfg['stereo_speckle_filter'])
        env['SUBPIX'] = '2'
        env['MEDIAN'] = '1'
        env['CENSUS_NCC_WIN'] = str(cfg['census_ncc_win'])
        # it is required that p2 > p1. The larger p1, p2, the smoother the disparity
        regularity_multiplier = cfg['stereo_regularity_multiplier']

        nb_dir = cfg['mgm_nb_directions']

        # increasing these numbers compensates the loss of regularity after incorporating LSD weights
        P1 = 12*regularity_multiplier   # penalizes disparity changes of 1 between neighbor pixels
        P2 = 48*regularity_multiplier  # penalizes disparity changes of more than 1
        conf = disp+'.confidence.tif'

        common.run(
            '{executable} '
            '-r {disp_min} -R {disp_max} '
            '-S 6 '
            '-s vfit '
            '-t census '
            '-O {nb_dir} '
            '-wl {wref} -wr {wsec} '
            '-P1 {P1} -P2 {P2} '
            '-confidence_consensusL {conf} '
            '{im1} {im2} {disp}'.format(
                executable='mgm_multi',
                disp_min=disp_min,
                disp_max=disp_max,
                nb_dir=nb_dir,
                wref=wref,
                wsec=wsec,
                P1=P1,
                P2=P2,
                conf=conf,
                im1=im1,
                im2=im2,
                disp=disp,
            ),
            env=env,
            timeout=timeout,
        )

        create_rejection_mask(disp, im1, im2, mask)


    if algo == 'mgm_multi':
        env['REMOVESMALLCC'] = str(cfg['stereo_speckle_filter'])
        env['MINDIFF'] = '1'
        env['CENSUS_NCC_WIN'] = str(cfg['census_ncc_win'])
        env['SUBPIX'] = '2'
        # it is required that p2 > p1. The larger p1, p2, the smoother the disparity
        regularity_multiplier = cfg['stereo_regularity_multiplier']

        nb_dir = cfg['mgm_nb_directions']

        P1 = 8*regularity_multiplier   # penalizes disparity changes of 1 between neighbor pixels
        P2 = 32*regularity_multiplier  # penalizes disparity changes of more than 1
        conf = '{}_confidence.tif'.format(os.path.splitext(disp)[0])

        common.run(
            '{executable} '
            '-r {disp_min} -R {disp_max} '
            '-S 6 '
            '-s vfit '
            '-t census '
            '-O {nb_dir} '
            '-P1 {P1} -P2 {P2} '
            '-confidence_consensusL {conf} '
            '{im1} {im2} {disp}'.format(
                executable='mgm_multi',
                disp_min=disp_min,
                disp_max=disp_max,
                nb_dir=nb_dir,
                P1=P1,
                P2=P2,
                conf=conf,
                im1=im1,
                im2=im2,
                disp=disp,
            ),
            env=env,
            timeout=timeout,
        )

        create_rejection_mask(disp, im1, im2, mask)

    if (algo == 'micmac'):
        # add micmac binaries to the PATH environment variable
        s2p_dir = os.path.dirname(os.path.dirname(os.path.realpath(os.path.abspath(__file__))))
        micmac_bin = os.path.join(s2p_dir, 'bin', 'micmac', 'bin')
        os.environ['PATH'] = os.environ['PATH'] + os.pathsep + micmac_bin

        # prepare micmac xml params file
        micmac_params = os.path.join(s2p_dir, '3rdparty', 'micmac_params.xml')
        work_dir = os.path.dirname(os.path.abspath(im1))
        common.run('cp {0} {1}'.format(micmac_params, work_dir))

        # run MICMAC
        common.run('MICMAC {0:s}'.format(os.path.join(work_dir, 'micmac_params.xml')))

        # copy output disp map
        micmac_disp = os.path.join(work_dir, 'MEC-EPI',
                                   'Px1_Num6_DeZoom1_LeChantier.tif')
        disp = os.path.join(work_dir, 'rectified_disp.tif')
        common.run('cp {0} {1}'.format(micmac_disp, disp))

        # compute mask by rejecting the 10% of pixels with lowest correlation score
        micmac_cost = os.path.join(work_dir, 'MEC-EPI',
                                   'Correl_LeChantier_Num_5.tif')
        mask = os.path.join(work_dir, 'rectified_mask.png')
        common.run(["plambda", micmac_cost, "x x%q10 < 0 255 if", "-o", mask])

