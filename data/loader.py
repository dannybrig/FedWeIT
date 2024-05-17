import os
import pdb
import glob
import numpy as np

from misc.utils import *

class DataLoader:
    """ Data Loader
        
    Loading data for the corresponding clients
    
    Created by:
        Wonyong Jeong (wyjeong@kaist.ac.kr)
    """
    def __init__(self, args):
        self.args = args
        self.base_dir = os.path.join(self.args.task_path, self.args.task) 
        self.did_to_dname = {
            0: 'cifar10',
            1: 'cifar100',
            2: 'mnist',
            3: 'svhn',
            4: 'fashion_mnist',
            5: 'traffic_sign',
            6: 'face_scrub',
            7: 'not_mnist',
        }

    def init_state(self, cid):
        self.state = {
            'client_id': cid,
            'tasks': []
        }
        self.load_tasks(cid)

    def load_state(self, cid):
        self.state = np_load(os.path.join(self.args.state_dir, '{}_data.npy'.format(cid))).item()

    def save_state(self):
        np_save(self.args.state_dir, '{}_data'.format(self.state['client_id']), self.state)

    def load_tasks(self, cid):
        if self.args.task in ['non_iid_50']:
            task_set = {
                0: ['cifar100_5', 'cifar100_13', 'face_scrub_0', 'cifar100_14', 'svhn_1', 'traffic_sign_0', 'not_mnist_1', 'cifar100_8', 'face_scrub_13', 'cifar100_4'],
                1: ['cifar100_2', 'traffic_sign_5', 'face_scrub_14', 'traffic_sign_4', 'not_mnist_0', 'mnist_0', 'face_scrub_2', 'face_scrub_15', 'cifar100_1', 'fashion_mnist_1'],
                2: ['face_scrub_11', 'svhn_0', 'face_scrub_10', 'face_scrub_6', 'face_scrub_7', 'cifar100_3', 'cifar100_10', 'mnist_1', 'face_scrub_1', 'traffic_sign_1'],
                3: ['fashion_mnist_0', 'cifar100_15', 'face_scrub_3', 'cifar10_1', 'cifar100_7', 'face_scrub_8', 'cifar10_0', 'face_scrub_9', 'cifar100_0', 'cifar100_6'],
                4: ['traffic_sign_7', 'face_scrub_5', 'traffic_sign_6', 'traffic_sign_3', 'traffic_sign_2','cifar100_12', 'cifar100_11', 'cifar100_9', 'face_scrub_12', 'face_scrub_4']
            }
            self.state['tasks'] = task_set[self.state['client_id']]

        elif self.args.task == 'mnist':
            task_set = {
                0: ['mnist_0_0', 'mnist_1_0', 'mnist_2_0', 'mnist_3_0', 'mnist_4_0'],
                1: ['mnist_0_1', 'mnist_1_1', 'mnist_2_1', 'mnist_3_1', 'mnist_4_1'], 
                2: ['mnist_0_2', 'mnist_1_2', 'mnist_2_2', 'mnist_3_2', 'mnist_4_2'],
                3: ['mnist_0_3', 'mnist_1_3', 'mnist_2_3', 'mnist_3_3', 'mnist_4_3'],
                4: ['mnist_0_4', 'mnist_1_4', 'mnist_2_4', 'mnist_3_4', 'mnist_4_4'],
                5: ['mnist_0_5', 'mnist_1_5', 'mnist_2_5', 'mnist_3_5', 'mnist_4_5'],
                6: ['mnist_0_6', 'mnist_1_6', 'mnist_2_6', 'mnist_3_6', 'mnist_4_6'],
                7: ['mnist_0_7', 'mnist_1_7', 'mnist_2_7', 'mnist_3_7', 'mnist_4_7'],
                8: ['mnist_0_8', 'mnist_1_8', 'mnist_2_8', 'mnist_3_8', 'mnist_4_8'],
                9: ['mnist_0_9', 'mnist_1_9', 'mnist_2_9', 'mnist_3_9', 'mnist_4_9'],
            }
            self.state['tasks'] = task_set[self.state['client_id']]

        elif self.args.task == 'cifar10':
            task_set = {
                0: ['cifar10_0_0', 'cifar10_1_0', 'cifar10_2_0', 'cifar10_3_0', 'cifar10_4_0'],
                1: ['cifar10_0_1', 'cifar10_1_1', 'cifar10_2_1', 'cifar10_3_1', 'cifar10_4_1'], 
                2: ['cifar10_0_2', 'cifar10_1_2', 'cifar10_2_2', 'cifar10_3_2', 'cifar10_4_2'],
                3: ['cifar10_0_3', 'cifar10_1_3', 'cifar10_2_3', 'cifar10_3_3', 'cifar10_4_3'],
                4: ['cifar10_0_4', 'cifar10_1_4', 'cifar10_2_4', 'cifar10_3_4', 'cifar10_4_4'],
                5: ['cifar10_0_5', 'cifar10_1_5', 'cifar10_2_5', 'cifar10_3_5', 'cifar10_4_5'],
                6: ['cifar10_0_6', 'cifar10_1_6', 'cifar10_2_6', 'cifar10_3_6', 'cifar10_4_6'],
                7: ['cifar10_0_7', 'cifar10_1_7', 'cifar10_2_7', 'cifar10_3_7', 'cifar10_4_7'],
                8: ['cifar10_0_8', 'cifar10_1_8', 'cifar10_2_8', 'cifar10_3_8', 'cifar10_4_8'],
                9: ['cifar10_0_9', 'cifar10_1_9', 'cifar10_2_9', 'cifar10_3_9', 'cifar10_4_9'],
            }
            self.state['tasks'] = task_set[self.state['client_id']]

        elif self.args.task == 'cifar100':
            task_set = {
                0: ['cifar100_0_0', 'cifar100_1_0', 'cifar100_2_0', 'cifar100_3_0', 'cifar100_4_0', 'cifar100_5_0', 'cifar100_6_0', 'cifar100_7_0', 'cifar100_8_0', 'cifar100_9_0'],
                1: ['cifar100_0_1', 'cifar100_1_1', 'cifar100_2_1', 'cifar100_3_1', 'cifar100_4_1', 'cifar100_5_1', 'cifar100_6_1', 'cifar100_7_1', 'cifar100_8_1', 'cifar100_9_1'], 
                2: ['cifar100_0_2', 'cifar100_1_2', 'cifar100_2_2', 'cifar100_3_2', 'cifar100_4_2', 'cifar100_5_2', 'cifar100_6_2', 'cifar100_7_2', 'cifar100_8_2', 'cifar100_9_2'],
                3: ['cifar100_0_3', 'cifar100_1_3', 'cifar100_2_3', 'cifar100_3_3', 'cifar100_4_3', 'cifar100_5_3', 'cifar100_6_3', 'cifar100_7_3', 'cifar100_8_3', 'cifar100_9_3'],
                4: ['cifar100_0_4', 'cifar100_1_4', 'cifar100_2_4', 'cifar100_3_4', 'cifar100_4_4', 'cifar100_5_4', 'cifar100_6_4', 'cifar100_7_4', 'cifar100_8_4', 'cifar100_9_4'],
                5: ['cifar100_0_5', 'cifar100_1_5', 'cifar100_2_5', 'cifar100_3_5', 'cifar100_4_5', 'cifar100_5_5', 'cifar100_6_5', 'cifar100_7_5', 'cifar100_8_5', 'cifar100_9_5'],
                6: ['cifar100_0_6', 'cifar100_1_6', 'cifar100_2_6', 'cifar100_3_6', 'cifar100_4_6', 'cifar100_5_6', 'cifar100_6_6', 'cifar100_7_6', 'cifar100_8_6', 'cifar100_9_6'],
                7: ['cifar100_0_7', 'cifar100_1_7', 'cifar100_2_7', 'cifar100_3_7', 'cifar100_4_7', 'cifar100_5_7', 'cifar100_6_7', 'cifar100_7_7', 'cifar100_8_7', 'cifar100_9_7'],
                8: ['cifar100_0_8', 'cifar100_1_8', 'cifar100_2_8', 'cifar100_3_8', 'cifar100_4_8', 'cifar100_5_8', 'cifar100_6_8', 'cifar100_7_8', 'cifar100_8_8', 'cifar100_9_8'],
                9: ['cifar100_0_9', 'cifar100_1_9', 'cifar100_2_9', 'cifar100_3_9', 'cifar100_4_9', 'cifar100_5_9', 'cifar100_6_9', 'cifar100_7_9', 'cifar100_8_9', 'cifar100_9_9'],
            }

        else:
            print('no correct task was given: {}'.format(self.args.task))
            os._exit(0)

    def get_train(self, task_id):
        return load_task(self.base_dir, self.state['tasks'][task_id]+'_train.npy').item()
    

    def get_valid(self, task_id):
        valid = load_task(self.base_dir, self.state['tasks'][task_id]+'_valid.npy').item()
        return valid['x_valid'], valid['y_valid']

    def get_test(self, task_id):
        x_test_list = []
        y_test_list = []
        for tid, task in enumerate(self.state['tasks']):
            if tid <= task_id:
                test = load_task(self.base_dir, task+'_test.npy').item()
                x_test_list.append(test['x_test'])
                y_test_list.append(test['y_test'])
        return x_test_list, y_test_list

    
