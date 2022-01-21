import os.path
from glob import glob
from shutil import rmtree

import sys

import numpy as np
import pandas as pd

sys.path.append('../')

from common.camera import world_to_camera, project_to_2d, image_coordinates
from utils.utils import wrap

output_filename = 'data_3d_mocap'
output_filename_2d = 'data_2d_mocap_gt'
subjects = ['S15', 'S16', 'S18', 'S19', 'S20', 'S23', 'S24', 'S25', 'S26', 'S28', 'S29', 'S30', 'S31', 'S32', 'S33']

if __name__ == '__main__':
    output = {}
    import pdb
    # pos_3d extract

    for subject in subjects :
        output[subject] = {}
        file_list = glob( 'tcr_final/' + subject + '*.csv')

        # 단일 csv 파일 처리
        for f in file_list :

            # action name 추출
            f_name = os.path.splitext(os.path.basename(f))[0]   # S15_Walking1
            tmp = subject + '_' # S15_
            action = f_name.replace(tmp, "")    #Walking1

            #---- 파일 가져오기 ----#
            df = pd.read_csv(f, skiprows=3 ) # f = 'tcr_final/S15_Walking2.csv'

            # data 구조 : 60s (7805,227),(7805,1) / 30s (4205,227),(4204, 1)
            if len(df.columns) == 1:
                df = pd.read_csv(f, delimiter = '\t', skiprows=3 )
                print(f, df.shape)
                df_ndarray = np.array(df.values)
                df_ndarray = df_ndarray[:,:227]
            else : df_ndarray = np.array(df.values)


            # 데이터 정보(프레임 num, joint 이름...) 행렬 삭제
            if df_ndarray.shape[0] < 7200 : # for 30s data
                df_ndarray = df_ndarray[2:3602, 2:]
            else : # for 60s data
                df_ndarray = df_ndarray[2:7202,2:]

            # frame 축소 (7200->3600 / 3600->1800)
            l = []
            for i in range(df_ndarray.shape[0]):
                if i % 2 != 0:
                    l.append(i)
            df_ndarray = np.delete(df_ndarray, l, axis=0) # shape (3600,225)


            # nan data 처리
            n_pd = pd.DataFrame(df_ndarray)
            n_fill = n_pd.fillna(method='ffill')
            filled_ndarray = np.array(n_fill)
            filled_ndarray = filled_ndarray.reshape(filled_ndarray.shape[0], -1, 3).astype('float32')  # shape (3600, 75, 3)

            # joint 배열 정렬
            '''
              (to)  (from)
                0 : joint[ 58 ] V_Mid_Hip
                1 : joint[ 48 ] V_R.Hip_JC
                2 : joint[ 54 ] V_R.Knee_JC
                3 : joint[ 56 ] V_R.Ankle_JC
                4 : joint[ 49 ] V_L.Hip_JC
                5 : joint[ 55 ] V_L.Knee_JC
                6 : joint[ 57 ] V_L.Ankle_JC
                7 : joint[ 47 ] V_Pelvis_Origin
                8 : joint[ 4 ] Clavical
                9 : joint[ 74 ] V_Head.Center
                10 : joint[ 14 ] L.Shoulder
                11 : joint[ 71 ] V_L.Elbow_JC
                12 : joint[ 69 ] V_L.Wrist_JC
                13 : joint[ 6 ] R.Shoulder
                14 : joint[ 70 ] V_R.Elbow_JC
                15 : joint[ 68 ] V_R.Wrist_JC
            '''
            joint_array = [58, 48, 54, 56, 49, 55, 57, 47, 4, 74, 14, 71, 69, 6, 70, 68]
            out_array = np.empty((filled_ndarray.shape[0], 16, 3)) # (num_frame, num_joints, dims)

            for i, j in enumerate(joint_array):
                if (j == 47) or (j == 4):
                    if j == 47:
                        b = 59
                    elif j == 14:
                        b = 5
                    temp = (filled_ndarray[:, j, :] + filled_ndarray[:, b, :]) / 2
                    out_array[:, i, :] = temp
                else:
                    out_array[:, i, :] = filled_ndarray[:, j, :]
            out_array = out_array/1000 # mm to meters (idk)
            print('\n\nshape : ', filled_ndarray.shape, ' -> shape : ', out_array.shape) #shape (3600,16,3)

            output[subject][action] = out_array.astype('float32')

            print(subject, action, output[subject][action].shape)
    print('Saving...')
    np.savez_compressed(output_filename, position_3d = output)
    print('Done')

    # project 3d pos to 2d
    print('\nComputing ground-truth 2D poses...')
    from common.mocap_dataset_edit import MocapCustomDataset
    dataset = MocapCustomDataset(output_filename + '.npz')
    output_2d_poses = {}
    for subject in dataset.subjects():
        output_2d_poses[subject] = {}

        for action in dataset[subject].keys():
            anim = dataset[subject][action]
            positions_2d = []

            cam = anim['cameras']
            pos_3d = world_to_camera(anim['positions'], R = cam['orientation'], t = cam['translation'])
            '''
            array([[-1.3488696 , -0.24857378,  4.79955   ],
                   [-1.3040178 , -0.23026037,  4.7247267 ],
                   [-1.3187338 ,  0.200634  ,  4.811812  ],
                   [-1.3509145 ,  0.5645275 ,  4.8504715 ],
                   [-1.3937213 , -0.2668872 ,  4.8743734 ],
                   [-1.4118214 ,  0.16523886,  4.955156  ],
                   [-1.4570156 ,  0.51063347,  5.0568824 ],
                   [-1.3552315 , -0.5618615 ,  4.710492  ],
                   [-1.3334893 , -0.77032995,  4.6701655 ],
                   [-1.3132353 , -0.9843581 ,  4.6633143 ],
                   [-1.4606302 , -0.8275404 ,  4.776492  ],
                   [-1.3543931 , -0.7487979 ,  5.0377254 ],
                   [-1.2520243 , -0.86449814,  4.8459625 ],
                   [-1.2744836 , -0.7376094 ,  4.504545  ],
                   [-1.1939089 , -0.42027903,  4.4811726 ],
                   [-1.1860805 , -0.17325258,  4.6190343 ]], dtype=float32)
            '''
            pos_2d = wrap(project_to_2d, True, pos_3d, cam['intrinsic'])
            '''
            (Pdb) pos_2d[0]
            array([[-0.63806653, -0.93154764],
                   [-0.6278276 , -0.925215  ],
                   [-0.6238672 , -0.73641086],
                   [-0.6317437 , -0.5811131 ],
                   [-0.64797705, -0.93767905],
                   [-0.6460362 , -0.75392294],
                   [-0.6516042 , -0.6132761 ],
                   [-0.6505569 , -1.0718033 ],
                   [-0.64498496, -1.1662765 ],
                   [-0.6355408 , -1.2612001 ],
                   [-0.68574554, -1.182777  ],
                   [-0.61151546, -1.133045  ],
                   [-0.5893152 , -1.1947832 ],
                   [-0.63974154, -1.1638776 ],
                   [-0.60774225, -1.0192407 ],
                   [-0.5886658 , -0.90189356]], dtype=float32)
            '''
            pos_2d_pixel_space = image_coordinates(pos_2d, w=cam['res_w'], h=cam['res_h'])
            '''
            (Pdb) pos_2d_pixel_space[0]
            array([[195.44407368, 456.96427345],
                   [200.97310424, 460.38389683],
                   [203.11170459, 562.33813763],
                   [198.85838628, 646.19892597],
                   [190.09239078, 453.65331173],
                   [191.14044785, 552.88161278],
                   [188.13374519, 628.83089304],
                   [188.69926214, 381.22620106],
                   [191.7081213 , 330.21071434],
                   [196.80797696, 278.95196199],
                   [169.69740987, 321.30039454],
                   [209.7816503 , 348.15572262],
                   [221.76980495, 314.81706619],
                   [194.53956842, 331.50609255],
                   [211.81918502, 409.61000204],
                   [222.12047696, 472.9774797 ]])
            '''
            positions_2d.append(pos_2d_pixel_space.astype('float32'))

            output_2d_poses[subject][action] = positions_2d

    print('Saving...')
    metadata = {
        'num_joints': dataset.skeleton().num_joints(),
        'keypoints_symmetry':[dataset.skeleton().joints_left(),dataset.skeleton().joints_right()]
    }
    np.savez_compressed(output_filename_2d, positions_2d = output_2d_poses, metadata = metadata)

    print('Done.')
