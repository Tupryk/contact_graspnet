import os
import sys

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))

import numpy as np
import config_utils
from contact_grasp_estimator import GraspEstimator

class GraspnetModel():
    def __init__(self, checkpoint_dir='./contact_graspnet/checkpoints/scene_test_2048_bs3_hor_sigma_001'):
        global_config = config_utils.load_config(checkpoint_dir, batch_size=1, arg_configs=[])

        self.grasp_estimator = GraspEstimator(global_config)
        self.grasp_estimator.build_network()

        saver = tf.train.Saver(save_relative_paths=True)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)

        self.grasp_estimator.load_weights(self.sess, saver, checkpoint_dir, mode='test')

    def predict(self, pc_full, local_regions=False, filter_grasps=False, forward_passes=1):
        pred_grasps_cam, scores, _, _ = self.grasp_estimator.predict_scene_grasps(self.sess, pc_full, pc_segments={}, 
                                                                                    local_regions=local_regions,
                                                                                    filter_grasps=filter_grasps,
                                                                                    forward_passes=forward_passes)
        grasp_index = np.argmax(scores[-1])
        grasp_pose = pred_grasps_cam[-1][grasp_index]
        return grasp_pose
