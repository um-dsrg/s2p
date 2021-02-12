"""
    conduct stereo matching based on trained model + a series of post-processing
"""


import os
#import util
import time
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime
from tqdm import tqdm
#from process_functional import *
import libmccnn.process_functional_Bi

from skimage import io as io #numpy version has to be ==1.15.0
from scipy.signal import medfilt2d

def median_cost_volume(cost_volume,kernel_size = 3):
    for i in range(len(cost_volume)):
        # Compute the median filter on the cost volume
        cost_volume[i,:,:] = medfilt2d(cost_volume[i,:,:],kernel_size)
    
    return cost_volume
        
        	
def compute_disparity_mccnn_basic(rect1, rect2, disp, disp_min, disp_max, resume):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    patch_size=11
    patch_height = patch_size
    patch_width = patch_size

    ####################
    # do matching
    # get data path
    left_path = rect1
    right_path = rect2

    # generate output path
    #out_img_path = disp
    #out_time_path = os.path.join(res_dir, out_time_file)
    #out_imgR_path = os.path.join(img_dir, out_imgR_file)

    # reading images
    left_image = io.imread(left_path).astype(np.float32)
    right_image = io.imread(right_path).astype(np.float32)
    left_image = (left_image - np.mean(left_image, axis=(0, 1))) / np.std(left_image, axis=(0, 1))
    right_image = (right_image - np.mean(right_image, axis=(0, 1))) / np.std(right_image, axis=(0, 1))
    left_image = np.expand_dims(left_image, axis=2)
    right_image = np.expand_dims(right_image, axis=2)
    print("{}: images read".format(datetime.now()))
    
    
    ####################
    # limit disparity bounds
    disp_min = [disp_min]
    disp_max = [disp_max]
    np.alltrue(len(disp_min) == len(disp_max))
    for dim in range(len(disp_min)):
        if disp_min[dim] is not None and disp_max[dim] is not None:
            image_size = left_image.shape
            if disp_max[dim] - disp_min[dim] > image_size[dim]:
                center = 0.5 * (disp_min[dim] + disp_max[dim])
                disp_min[dim] = int(center - 0.5 * image_size[dim])
                disp_max[dim] = int(center + 0.5 * image_size[dim])
        # round disparity bounds
        if disp_min[dim] is not None:
            disp_min[dim] = int(np.floor(disp_min[dim]))
        if disp_max is not None:
            disp_max[dim] = int(np.ceil(disp_max[dim]))
    disp_min = disp_min[0]
    disp_max = disp_max[0]
    
    print((disp_min, disp_max))
    # start timer for time file
    stTime = time.time()
    
    # Derive the dimensions of the two images
    height, width = left_image.shape[:2]

    # Compute the left and right features
    featuresl, featuresr = libmccnn.process_functional_Bi.compute_features(left_image, right_image, None, None, patch_height, patch_width, resume,1)
    #print("{}: features computed".format(datetime.now()))
    
    # Construct the left and right cost volumes
    #print('Construct the left and right cost volumes ...')
    left_cost_volume, right_cost_volume = libmccnn.process_functional_Bi.compute_cost_volume(featuresl,featuresr,disp_min, disp_max)

    # Apply a median filter to the cost volumes
    left_cost_volume = median_cost_volume(left_cost_volume)
    right_cost_volume = median_cost_volume(right_cost_volume)
    
    # Compute the left disparity map
    print('Compute the left disparity map')
    left_disparity_map = libmccnn.process_functional_Bi.disparity_selection(left_cost_volume, np.arange(disp_min, disp_max+1,1))
    
    # Compute the right disparity map
    print('Compute the right disparity map')
    #right_disparity_map = -libmccnn.process_functional_Bi.disparity_selection(right_cost_volume, np.arange(disp_min, disp_max+1,1))
    right_disparity_map = libmccnn.process_functional_Bi.disparity_selection(right_cost_volume, np.arange(disp_min, disp_max+1,1))
    
    # Estimate the disparity map aligned with the left view
    print('Compute the left right consistency')
    disparity_map = libmccnn.process_functional_Bi.left_right_consistency(left_disparity_map, right_disparity_map)

    # Save the negative disparity since this correlates more with the disparity map
    disparity_map = -disparity_map
    
    # save as pgm and pfm
    io.imsave(disp, disparity_map)
'''
def compute_disparity_mccnn(rect1, rect2, disp, disp_min,disp_max):
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    patch_size=11
    patch_height = patch_size
    patch_width = patch_size

    ####################
    # do matching
    # get data path
    left_path = rect1
    right_path = rect2

    # generate output path
    #out_img_path = disp
    #out_time_path = os.path.join(res_dir, out_time_file)
    #out_imgR_path = os.path.join(img_dir, out_imgR_file)

    # reading images
    left_image = io.imread(left_path).astype(np.float32)
    right_image = io.imread(right_path).astype(np.float32)
    left_image = (left_image - np.mean(left_image, axis=(0, 1))) / np.std(left_image, axis=(0, 1))
    right_image = (right_image - np.mean(right_image, axis=(0, 1))) / np.std(right_image, axis=(0, 1))
    left_image = np.expand_dims(left_image, axis=2)
    right_image = np.expand_dims(right_image, axis=2)
    print("{}: images read".format(datetime.now()))

    ####################
    # limit disparity bounds
    disp_min = [disp_min]
    disp_max = [disp_max]
    np.alltrue(len(disp_min) == len(disp_max))
    for dim in range(len(disp_min)):
        if disp_min[dim] is not None and disp_max[dim] is not None:
            image_size = left_image.shape
            if disp_max[dim] - disp_min[dim] > image_size[dim]:
                center = 0.5 * (disp_min[dim] + disp_max[dim])
                disp_min[dim] = int(center - 0.5 * image_size[dim])
                disp_max[dim] = int(center + 0.5 * image_size[dim])
        # round disparity bounds
        if disp_min[dim] is not None:
            disp_min[dim] = int(np.floor(disp_min[dim]))
        if disp_max is not None:
            disp_max[dim] = int(np.ceil(disp_max[dim]))
    disp_min = disp_min[0]
    disp_max = disp_max[0]

    # start timer for time file
    stTime = time.time()

    # compute features
    left_feature, right_feature = process_functional_Bi.compute_features(left_image, right_image, patch_height, patch_width, args.resume)
    print(left_feature.shape)
    print("{}: features computed".format(datetime.now()))

    # form cost-volume
    if disp_min < 0:
        left_cost_volume, right_cost_volume = process_functional_Bi.compute_cost_volume_Bi(left_feature, right_feature, disp_min, disp_max)
        #left_cost_volume, right_cost_volume = process_functional_Bi.compute_cost_volume_Bi(left_feature, right_feature, -27, 24)
    else:
        left_cost_volume, right_cost_volume = process_functional_Bi.compute_cost_volume(left_feature, right_feature, disp_max)
    print("{}: cost-volume computed".format(datetime.now()))

    postprocess_flag = args.postProcess

    if postprocess_flag:
        # cost-volume aggregation
        print("{}: begin cost-volume aggregation. This could take long".format(datetime.now()))
        left_cost_volume, right_cost_volume = process_functional_Bi.cost_volume_aggregation(left_image, right_image, left_cost_volume,
                                                                  right_cost_volume, \
                                                                  args.cbca_intensity, args.cbca_distance,
                                                                  args.cbca_num_iterations1)
        print("{}: cost-volume aggregated".format(datetime.now()))

        # semi-global matching
        print("{}: begin semi-global matching. This could take long".format(datetime.now()))
        left_cost_volume, right_cost_volume = process_functional_Bi.SGM_average(left_cost_volume, right_cost_volume, left_image, right_image, \
                                                      args.sgm_P1, args.sgm_P2, args.sgm_Q1, args.sgm_Q2,
                                                      args.sgm_D, args.sgm_V)
        print("{}: semi-global matched".format(datetime.now()))

        # cost-volume aggregation afterhand
        print("{}: begin cost-volume aggregation. This could take long".format(datetime.now()))
        left_cost_volume, right_cost_volume = process_functional_Bi.cost_volume_aggregation(left_image, right_image, left_cost_volume, right_cost_volume, \
                                                                  args.cbca_intensity, args.cbca_distance, args.cbca_num_iterations2)
        print("{}: cost-volume aggregated".format(datetime.now()))

    # disparity map making
    left_disparity_map, right_disparity_map = process_functional_Bi.disparity_prediction(left_cost_volume, right_cost_volume)
    print("{}: disparity predicted".format(datetime.now()))

    if postprocess_flag:
        # interpolation
        ndisp = disp_max - disp_min
        left_disparity_map = process_functional_Bi.interpolation(left_disparity_map, right_disparity_map, ndisp)
        print("{}: disparity interpolated".format(datetime.now()))

        # subpixel enhancement
        left_disparity_map = process_functional_Bi.subpixel_enhance(left_disparity_map, left_cost_volume)
        print("{}: subpixel enhanced".format(datetime.now()))

        # refinement
        # 5*5 median filter
        left_disparity_map = process_functional_Bi.median_filter(left_disparity_map, 5, 5)

        # bilateral filter
        left_disparity_map = process_functional_Bi.bilateral_filter(left_image, left_disparity_map, 5, 5, 0, args.blur_sigma,
                                          args.blur_threshold)
    print("{}: refined".format(datetime.now()))


    # revise mchen 20200606
    if disp_min < 0:
        left_disparity_map += disp_min
        right_disparity_map += disp_min
    left_disparity_map = -left_disparity_map
    right_disparity_map = -right_disparity_map

    # end timer
    endTime = time.time()

    # save as pgm and pfm
    io.imsave(disp, left_disparity_map)
    #util.saveDisparity(left_disparity_map, out_imgL_path)
    #util.writePfm(left_disparity_map, out_path)
    #util.saveTimeFile(endTime - stTime, out_time_path)
    print("{}: saved".format(datetime.now()))
'''
