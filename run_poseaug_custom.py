from __future__ import print_function, absolute_import, division

import os
import random

import torch.backends.cudnn as cudnn

from common.camera import *
from common.render import *

from function_baseline.model_pos_preparation import model_pos_preparation
from function_poseaug.config import get_parse_args
from function_poseaug.data_preparation import data_preparation
from function_poseaug.data_preparation_custom import data_preparation_custom
import pdb
'''
This code is used to train PoseAug model 
1. Simple Baseline
2. VideoPose
3. SemGCN
4. ST-GCN
'''


def main(args):
    print('==> Using settings {}'.format(args))
    device = torch.device("cuda")
    
    print('==> Loading dataset...')
    if args.dataset == 'custom':
        data_dict = data_preparation_custom(args)
    else: data_dict = data_preparation(args)


    print("==> Creating model...")
    model_pos = model_pos_preparation(args, data_dict['dataset'], device)

    print("==> Loading checkpoint '{}'".format(args.evaluate))
    ckpt = torch.load(args.evaluate)
    print('This model was trained for {} epochs'.format(ckpt['epoch']))
    try:
        model_pos.load_state_dict(ckpt['state_dict'])
    except:
        model_pos.load_state_dict(ckpt['model_pos'])


    def evaluate(data_loader, model_pos_eval, device, return_predictions=False):
        print("==> Prediction")
        model_pos_eval.eval()

        for i, poses_2d in enumerate(data_loader):
            inputs_2d = poses_2d.to(device)
            num_poses = inputs_2d.size(0)

            with torch.no_grad():
                outputs_3d = model_pos_eval(inputs_2d.view(num_poses, -1)).view(num_poses, -1, 3).cpu()
                pdb.set_trace()
                #outputs_3d = outputs_3d[:, :, :] - outputs_3d[:, :1, :]
                if return_predictions:
                    outputs_3d = outputs_3d.cpu()
                    return outputs_3d.squeeze(0).numpy()

    if args.render:

        print('==> Rendering...')
        poses_2d = data_dict['keypoints'][args.viz_subject][args.viz_action][args.viz_camera]
        input_keypoints = poses_2d.copy()

        prediction = evaluate(data_dict['valid_loader'], model_pos, device, return_predictions=True)

        # Invert camera transformation
        cam = data_dict['dataset'].cameras()[args.viz_subject][args.viz_camera]
        prediction = camera_to_world(prediction, R=cam['orientation'], t=0)
        # added a constraint to force the height of the lowest joint to 0, so that the hip joint isn't center...
        prediction[:, :, 2] -= np.min(prediction[:, :, 2], axis=1, keepdims=True)

        anim_output = {'Regression': prediction}
        input_keypoints = image_coordinates(input_keypoints[..., :2], w=cam['res_w'], h=cam['res_h'])

        
        render_animation(input_keypoints, data_dict['keypoints_metadata'], anim_output, data_dict['dataset'].skeleton(), data_dict['dataset'].fps(), args.viz_bitrate, cam['azimuth'],
        args.viz_output, limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
        input_video_path=args.viz_video, viewport=(cam['res_w'], cam['res_h']))


if __name__ == '__main__':
    args = get_parse_args()

    # fix random
    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    # copy from #https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = True

    main(args)
