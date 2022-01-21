from __future__ import absolute_import

import numpy as np
import copy
from common.skeleton import Skeleton
from common.camera import normalize_screen_coordinates

# about only 15 joint ( used in poseaug )
mocap_skeleton = Skeleton( parents=[-1,  0,  1,  2,  0,  4,  5,  0,  7,  8,  8, 10, 11,  8, 13, 14],
                           joints_left=[4, 5, 6, 10, 11, 12],
                           joints_right=[1, 2, 3, 13, 14, 15])
'''
# Video resolution
metadata = {
    'w': im.shape[1],
    'h': im.shape[0],
}
'''
mocap_cameras_params = {
    'res_w':1080,
    'res_h':1920,

    # Dummy camera parameters (taken from Human3.6M), Only for visualization purpose
    'orientation': [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088],
    'translation': [1841.1070556640625, 4955.28466796875, 1563.4454345703125],

    #intrinsic_cam
    'center': [512.54150390625, 515.4514770507812],
    'focal_length': [1145.0494384765625, 1143.7811279296875],
    'radial_distortion': [-0.20709891617298126, 0.24777518212795258, -0.0030751503072679043],
    'tangential_distortion': [-0.0009756988729350269, -0.00142447161488235],
    'azimuth': 70,  # Only used for visualization
}

class MocapCustomDataset(object):
    def __init__(self, path, fps=None):
        self._skeleton = mocap_skeleton
        self._fps = fps
        self._data = {}

        cam = {}
        cam.update(mocap_cameras_params)
        cam['orientation'] = np.array(cam['orientation'], dtype='float32')

        #Normalize camera frame
        for i, j in enumerate(['center', 'focal_length', 'radial_distortion', 'tangential_distortion']) :
            cam[j] = np.array(cam[j], dtype = 'float32')

        cam['center'] = normalize_screen_coordinates(cam['center'], w=cam['res_w'], h=cam['res_h']).astype('float32')
        cam['focal_length'] = cam['focal_length'] / cam['res_w'] * 2.0
        cam['translation'] = np.array(cam['translation'], dtype='float32')
        cam['translation'] = cam['translation'] / 1000  # mm to meters
        cam['intrinsic'] = np.concatenate((cam['focal_length'],
                                           cam['center'],
                                           cam['radial_distortion'],
                                           cam['tangential_distortion']))

        self._cameras = cam

        data = np.load(path, allow_pickle=True)['position_3d'].item()

        self._data = {}

        for subject, actions in data.items():
            self._data[subject] = {}
            for action_name, positions in actions.items():
                self._data[subject][action_name] = {
                    'positions':positions,
                    'cameras': self._cameras
                }

    def remove_joints(self, joints_to_remove):
        kept_joints = self._skeleton.remove_joints(joints_to_remove)
        for subject in self._data.keys():
            for action in self._data[subject].keys():
                s = self._data[subject][action]
                #add
                if 'positions' in s :
                    s['positions'] = s['positions'][:, kept_joints]
                #s['positions'] = s['positions'][:, kept_joints]

    def __getitem__(self, key):
        return self._data[key]

    def subjects(self):
        return self._data.keys()

    def fps(self):
        return self._fps

    def skeleton(self):
        return self._skeleton

    def cameras(self):
        return self._cameras

    def define_actions(self, action = None):
        all_actions = ['Walking1',
                       'Walking2',
                       'WalkingAround1',
                       'WalkingAround2',
                       'WalkingOnTreadmil1',
                       'WalkingOnTreadmil2',]

        if action is None:
            return all_actions

        if action not in all_actions:
            raise (ValueError, "Undefined action: {}".format(action))

        return [action]
