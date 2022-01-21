from __future__ import print_function, absolute_import, division

import argparse
import os
import zipfile
import numpy as np
from glob import glob
from shutil import rmtree

import sys

sys.path.append('../')

from common.h36m_dataset import Human36mDataset
from common.camera import world_to_camera, project_to_2d, image_coordinates
from utils.utils import wrap

output_filename = 'data_3d_h36m'
output_filename_2d = 'data_2d_h36m_gt'
subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']

if __name__ == '__main__':
    if os.path.basename(os.getcwd()) != 'data':
        print('This script must be launched from the "data" directory')
        exit(0)

    parser = argparse.ArgumentParser(description='Human3.6M dataset downloader/converter')

    # Default: convert dataset preprocessed by Martinez et al. in https://github.com/una-dinosauria/3d-pose-baseline
    parser.add_argument('--from-archive', default='', type=str, metavar='PATH', help='convert preprocessed dataset')

    # Alternatively, convert dataset from original source (the Human3.6M dataset path must be specified manually)
    parser.add_argument('--from-source', default='', type=str, metavar='PATH', help='convert original dataset')

    args = parser.parse_args()

    if args.from_archive and args.from_source:
        print('Please specify only one argument')
        exit(0)

    #if os.path.exists(output_filename + '.npz'):
        #print('The dataset already exists at', output_filename + '.npz')
        #exit(0)

    if args.from_archive:
        print('Extracting Human3.6M dataset from', args.from_archive)
        with zipfile.ZipFile(args.from_archive, 'r') as archive:
            archive.extractall()

        print('Converting...')
        output = {}

        import h5py
                        # ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
        for subject in subjects:
            output[subject] = {} # outputp['S1']
            file_list = glob('h36m/' + subject + '/MyPoses/3D_positions/*.h5') # 'h36m/S1/MyPose/3D_positions/*.h5
            assert len(file_list) == 30, "Expected 30 files for subject " + subject + ", got " + str(len(file_list))

            for f in file_list:
                action = os.path.splitext(os.path.basename(f))[0]  #Directions ...

                if subject == 'S11' and action == 'Directions':
                    continue  # Discard corrupted video
                with h5py.File(f) as hf:
                    positions = hf['3D_positions'].value.reshape(32, 3, -1).transpose(2, 0, 1)
                    '''
                    positions
                        array([[[ -141.33900452,    69.12960052,   933.91900635],
                                [ -273.72940063,    58.47375107,   939.79827881],
                                [ -242.15586853,   122.25301361,   502.65875244],
                              ...
                    positions.shape : (3134,32,3)
                    type(positions) : numpy
                    '''
                    positions /= 1000  # Meters instead of millimeters
                    '''
                    array([[-0.141339  ,  0.0691296 ,  0.93391901],
                           [-0.2737294 ,  0.05847375,  0.93979828],
                           [-0.24215587,  0.12225301,  0.50265875],
                           [-0.21795303,  0.20169511,  0.056109  ],
                    '''
                    output[subject][action] = positions.astype('float32')
                    # output[S1][Walking]
        print('Saving...')
        #np.savez_compressed(output_filename, positions_3d=output)

        print('Cleaning up...')
        #rmtree('h36m')

        print('Done.')

    elif args.from_source:
        print('Converting original Human3.6M dataset from', args.from_source)
        output = {}

        from scipy.io import loadmat

        for subject in subjects:
            output[subject] = {}
            file_list = glob(args.from_source + '/' + subject + '/MyPoseFeatures/D3_Positions/*.cdf.mat')
            assert len(file_list) == 30, "Expected 30 files for subject " + subject + ", got " + str(len(file_list))
            for f in file_list:
                file_name = os.path.splitext(os.path.splitext(os.path.basename(f))[0])[0]


                if subject == 'S11' and action == 'Directions':
                    continue  # Discard corrupted video

                # Use consistent naming convention
                canonical_name = action.replace('TakingPhoto', 'Photo').replace('WalkingDog', 'WalkDog')

                hf = loadmat(f)
                positions = hf['data'][0, 0].reshape(-1, 32, 3)
                positions /= 1000  # Meters instead of millimeters

                output[subject][canonical_name] = positions.astype('float32')

        print('Saving...')
        np.savez_compressed(output_filename, positions_3d=output)

        print('Done.')

    else:
        print('Please specify the dataset source')
        exit(0)

    import pdb;

    # Create 2D pose file
    print('')
    print('Computing ground-truth 2D poses...')
    dataset = Human36mDataset(output_filename + '.npz')
    output_2d_poses = {}
    for subject in dataset.subjects():
        output_2d_poses[subject] = {}
        for action in dataset[subject].keys():
            anim = dataset[subject][action]
            '''
            anim.keys()
            dict_keys(['positions', 'cameras']) 
            
            anim['positions']
            array([[[-0.141339  ,  0.0691296 ,  0.933919  ],
                    [-0.2737294 ,  0.05847375,  0.9397983 ],
                    [-0.24215586,  0.12225302,  0.5026587 ],
                    ...,
            anim['positions'].shape
            (3134, 16, 3)

            anim['cameras'][0].keys()
            dict_keys(['orientation', 'translation', 'id', 'center', 'focal_length', 'radial_distortion', 'tangential_distortion', 'res_w', 'res_h', 'azimuth', 'intrinsic'])

            '''
            positions_2d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                '''
                pos_3d
                array([[[-1.6485167e-01, -3.6517429e-01,  5.2953162e+00],
                        [-4.7796726e-02, -3.7969542e-01,  5.3566575e+00],
                        [-6.2326193e-02,  6.2815189e-02,  5.3680182e+00],
                        ...,
                        
                pos_3d.shape
                (3134, 16, 3)
                '''
                pos_2d = wrap(project_to_2d, True, pos_3d, cam['intrinsic'])
                pdb.set_trace()
                '''
                cam.keys()
                dict_keys(['orientation', 'translation', 'id', 'center', 'focal_length', 'radial_distortion', 
                        'tangential_distortion', 'res_w', 'res_h', 'azimuth', 'intrinsic'])
                cam['intrinsic']
                array([ 2.2900989e+00,  2.2875624e+00,  2.5083065e-02,  2.8902981e-02,
                       -2.0709892e-01,  2.4777518e-01, -3.0751503e-03, -9.7569887e-04,
                       -1.4244716e-03], dtype=float32)
                '''
                '''
                pos_2d
                array([[[-0.0461494 , -0.12870453],
                            [ 0.00465665, -0.13311014],
                            [-0.00150545,  0.05566892],
                            ...,

                pos_2d.shape
                (3134, 16, 2)
                '''
                pos_2d_pixel_space = image_coordinates(pos_2d, w=cam['res_w'], h=cam['res_h'])
                positions_2d.append(pos_2d_pixel_space.astype('float32'))
            output_2d_poses[subject][action] = positions_2d

    print('Saving...')
    metadata = {
        'num_joints': dataset.skeleton().num_joints(),
        'keypoints_symmetry': [dataset.skeleton().joints_left(), dataset.skeleton().joints_right()]
    }
    np.savez_compressed(output_filename_2d, positions_2d=output_2d_poses, metadata=metadata)

    print('Done.')
