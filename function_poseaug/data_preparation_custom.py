from __future__ import print_function, absolute_import, division

import os.path as path
import numpy as np
from torch.utils.data import DataLoader

from common.data_loader import PoseCustomDataSet
from utils.data_utils import fetch

'''
this code is used for prepare data loader
'''

def data_preparation_custom(args):
    """
    load the custom h36m dataset
    generate data loader for training posenet, poseaug, and cross-data evaluation
    """
    if args.dataset == 'custom':
        from common.custom_dataset import CustomDataset
        dataset = CustomDataset('data/data_2d_'+ args.dataset + '_' + args.keypoints + '.npz' )

        if not args.render:
            from common.h36m_dataset import TEST_SUBJECTS
            subjects_test = TEST_SUBJECTS
        else :
            subjects_test = [args.viz_subject]

        print('==> Loading 2D detections...')

        data_path = path.join('data', 'data_2d_' + args.dataset + '_' + args.keypoints + '.npz')

        keypoints = np.load(data_path, allow_pickle=True) # type : numpy zip (.npz)

        keypoints_metadata = keypoints['metadata'].item()
        keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
        kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
        joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
        keypoints = keypoints['positions_2d'].item()

        for subject in keypoints.keys():
            for action in keypoints[subject]:
                for cam_idx, kps in enumerate(keypoints[subject][action]):
                    # Normalize camera frame
                    cam = dataset.cameras()[subject][cam_idx]
                    from common.camera import normalize_screen_coordinates
                    kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                    keypoints[subject][action][cam_idx] = kps
    else:
        raise KeyError('Invalid dataset')

    action_filter = None if args.actions == '*' else args.actions.split(',')
    if action_filter is not None:
        action_filter = map(lambda x: dataset.define_actions(x)[0], action_filter)
        print('==> Selected actions: {}'.format(action_filter))

    stride = args.downsample

    ############################################
    # general 2D-3D pair dataset
    ############################################
    poses_valid, poses_valid_2d, actions_valid, cams_valid = fetch(subjects_test, dataset, keypoints, action_filter,
                                                                   stride)
    valid_loader = DataLoader(PoseCustomDataSet(poses_valid_2d),
                                  batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return {
        'dataset': dataset,

        'poses_valid': poses_valid,
        'poses_valid_2d':poses_valid_2d,
        'actions_valid': actions_valid,
        'cams_valid': cams_valid,

        'valid_loader': valid_loader,


        'action_filter': action_filter,
        'subjects_test': subjects_test,

        'keypoints': keypoints,
        'keypoints_metadata' :keypoints_metadata,
        'keypoints_symmetry': keypoints_symmetry,
        'kps_left' : kps_left,
        'kps_right': kps_right,
        'joints_left': joints_left,
        'joints_right' : joints_right
    }
