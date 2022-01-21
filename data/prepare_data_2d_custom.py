# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
from glob import glob
import os
import sys

import argparse

output_prefix_2d = 'data_2d_custom_'

def decode(filename):
    # Latin1 encoding because Detectron runs on Python 2.7
    print('Processing {}'.format(filename))
    data = np.load(filename, encoding='latin1', allow_pickle=True)
    bb = data['boxes']
    kp = data['keypoints']
    metadata = data['metadata'].item()

    results_bb = []
    results_kp = []
    for i in range(len(bb)):
        if len(bb[i][1]) == 0 or len(kp[i][1]) == 0:
            # No bbox/keypoints detected for this frame -> will be interpolated
            results_bb.append(np.full(4, np.nan, dtype=np.float32)) # 4 bounding box coordinates
            results_kp.append(np.full((17, 4), np.nan, dtype=np.float32))
            continue
        best_match = np.argmax(bb[i][1][:, 4])
        best_bb = bb[i][1][best_match, :4]
        best_kp = kp[i][1][best_match].T.copy() # shape: (17,4)
        results_bb.append(best_bb)
        results_kp.append(best_kp)
    
    bb = np.array(results_bb, dtype=np.float32)
    kp = np.array(results_kp, dtype=np.float32) # shape : (750,17,4)
    kp = kp[:, :, :2] # Extract (x, y)

    # Fix missing bboxes/keypoints by linear interpolation
    mask = ~np.isnan(bb[:, 0])
    indices = np.arange(len(bb))
    for i in range(4):
        bb[:, i] = np.interp(indices, indices[mask], bb[mask, i])
    
    for i in range(17):
        for j in range(2):
            kp[:, i, j] = np.interp(indices, indices[mask], kp[mask, i, j])

    out_kp = coco_2_h36m(kp)

    print('{} total frames processed'.format(len(bb)))
    print('{} frames were interpolated'.format(np.sum(~mask)))
    print('----------')
    
    return [{
        'start_frame': 0, # Inclusive
        'end_frame': len(kp), # Exclusive
        'bounding_boxes': bb,
        'keypoints': out_kp,
    }], metadata

def coco_2_h36m(kp):
    # -----keypoint rearrange --------- #
    # kp (coco format) to new_kp (h36m format)
    (a, b, c) = kp.shape
    new_kp = np.zeros((a, b - 1, c))

    new_kp[:, 0, :] = (kp[:, 11, :] + kp[:, 12, :]) / 2  # hip = ([11]Lhip + [12]Rhip) /2

    new_kp[:, 1, :] = kp[:, 12, :]  # Rhip
    new_kp[:, 2, :] = kp[:, 14, :]  # Rknee
    new_kp[:, 3, :] = kp[:, 16, :]  # Rfoot

    new_kp[:, 4, :] = kp[:, 11, :]  # Lhip
    new_kp[:, 5, :] = kp[:, 13, :]  # Lknee
    new_kp[:, 6, :] = kp[:, 15, :]  # Lfoot
    # ---------
    new_kp[:, 8, :] = (kp[:, 5, :] + kp[:, 6, :]) / 2  # Thorax = ([5]Rshoulder+ [6]Lshoulder) / 2
    new_kp[:, 7, :] = (new_kp[:, 0, :] + new_kp[:, 8, :]) / 2  # Spine = (throat + hip)/2  ...using new_kp
    new_kp[:, 9, :] = kp[:, 0, :] # Head
    # ---------
    new_kp[:, 10, :] = kp[:, 5, :]  # Lshoulder
    new_kp[:, 11, :] = kp[:, 7, :]  # Lelbow
    new_kp[:, 12, :] = kp[:, 9, :]  # Lwrist

    new_kp[:, 13, :] = kp[:, 6, :]  # Rshoulder
    new_kp[:, 14, :] = kp[:, 8, :]  # Relbow
    new_kp[:, 15, :] = kp[:, 10, :]  # Rwrist

    # smooth 2d keypoints for less jitter
    from scipy.signal import savgol_filter

    tmp = np.zeros(new_kp.shape)

    _, joint, dim = new_kp.shape
    # filtering one joints(x/y) within all frame at once
    for d in range(0, dim):
        for j in range(0, joint):
            tmp[:, j, d] = savgol_filter(new_kp[:, j, d], 11, 2, mode = 'nearest')

    return tmp

if __name__ == '__main__':
    if os.path.basename(os.getcwd()) != 'data':
        print('This script must be launched from the "data" directory')
        exit(0)
        
    parser = argparse.ArgumentParser(description='Custom dataset creator')
    parser.add_argument('-i', '--input', type=str, default='', metavar='PATH', help='detections directory')
    parser.add_argument('-o', '--output', type=str, default='', metavar='PATH', help='output suffix for 2D detections')
    args = parser.parse_args()
    
    if not args.input:
        print('Please specify the input directory')
        exit(0)
        
    if not args.output:
        print('Please specify an output suffix (e.g. detectron_pt_coco)')
        exit(0)

    print('Parsing 2D detections from', args.input)

    metadata = {'layout_name': 'h36m',
                'num_joints': 16,
                'keypoints_symmetry': [[4, 5, 6, 10, 11, 12], [1, 2, 3, 13, 14, 15]]
                }
    metadata['video_metadata'] = {}

    output = {}
    file_list = glob(args.input + '/*.npz')
    
    for f in file_list:
        canonical_name = os.path.splitext(os.path.basename(f))[0]
        data, video_metadata = decode(f)
        output[canonical_name] = {}
        output[canonical_name]['custom'] = [data[0]['keypoints'].astype('float32')]
        metadata['video_metadata'][canonical_name] = video_metadata

    print('Saving...')
    np.savez_compressed(output_prefix_2d + args.output, positions_2d=output, metadata=metadata)
    print('Done.')
