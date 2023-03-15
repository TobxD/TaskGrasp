import os
import copy
import sys
import pickle
import time
import os.path as osp
import shlex
import shutil
import subprocess
import argparse

#import lmdb
#import msgpack_numpy
import numpy as np
import torch
from torchvision import transforms
import torch.utils.data as data
import tqdm
from collections import defaultdict

BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, '../../'))
sys.path.append(os.path.join(BASE_DIR, '../'))
from utils.splits import get_split_data, parse_line, get_ot_pairs_taskgrasp
from visualize import draw_scene, get_gripper_control_points
from geometry_utils import regularize_pc_point_count
from data.data_specification import TASKS
import data.data_utils as d_utils

def pc_normalize(pc, grasp=None, pc_scaling=True):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    if grasp is not None:
        grasp[:3, 3] -= centroid

    if pc_scaling:
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))

        pc = np.concatenate([pc, np.ones([pc.shape[0], 1])], axis=1)
        scale_transform = np.diag([1 / m, 1 / m, 1 / m, 1])
        pc = np.matmul(scale_transform, pc.T).T
        pc = pc[:, :3]
        if grasp is not None:
            grasp = np.matmul(scale_transform, grasp)
    if grasp is not None:
        return pc, grasp
    else:
        return pc


def get_task1_hits(object_task_pairs, num_grasps=25):
    candidates = object_task_pairs['False'] + object_task_pairs['Weak False']
    lines = []
    label = -1  # All grasps are negatives
    for ot in candidates:
        for grasp_idx in range(num_grasps):
            obj, task = ot.split('-')
            line = "{}-{}-{}:{}\n".format(obj, str(grasp_idx), task, label)
            lines.append(line)
    return lines

def visualize_pc(object_pc, grasp_pc, label):
    col = [0, 255, 0] if label==1 else [255, 0, 0]
    grasp_color = torch.Tensor(col).double().repeat((grasp_pc.shape[0], 1))
    all_pc = torch.cat(
        [object_pc.double(), torch.cat([grasp_pc.double(), grasp_color.double()], dim=1)],
        dim=0
    )
    draw_scene(pc=all_pc)


class SGNTaskGrasp(data.Dataset):
    def __init__(
            self,
            num_points,
            transforms=None,
            train=0,
            download=True,
            base_dir=None,
            folder_dir='',
            normal=True,
            tasks=None,
            map_obj2class=None,
            class_list=None,
            split_mode=None,
            split_idx=0,
            split_version=0,
            pc_scaling=True,
            use_task1_grasps=True):
        """
        Args:
            num_points: Number of points in point cloud (used to downsample data to a fixed number)
            transforms: Used for data augmentation during training
            train: 1 for train, 0 for test, 2 for validation
            base_dir: location of dataset
            folder_dir: name of dataset
            tasks: list of tasks
            class_list: list of object classes
            map_obj2class: dictionary mapping dataset object to corresponding WordNet class
            split_mode: Choose between held-out instance ('i'), tasks ('t') and classes ('o')
            split_version: Choose 1, 0 is deprecated
            split_idx: For each split mode, there are 4 cross validation splits (choose between 0-3)
            pc_scaling: True if you want to scale the point cloud by the standard deviation
            include_reverse_relations: True since we are modelling a undirected graph
            use_task1_grasps: True if you want to include the grasps from the object-task pairs
                rejected in Stage 1 (and add these grasps are negative samples)

            Deprecated args (not used anymore): normal, download
        """
        super().__init__()

        self._pc_scaling = pc_scaling
        self._split_mode = split_mode
        self._split_idx = split_idx
        self._split_version = split_version
        self._num_points = num_points
        self._transforms = transforms
        self._tasks = tasks
        self._num_tasks = len(self._tasks)

        self._train = train
        self._map_obj2class = map_obj2class
        data_dir = os.path.join(base_dir, folder_dir, "scans")

        data_txt_splits = {
            0: 'test_split.txt',
            1: 'train_split.txt',
            2: 'val_split.txt'}
        if train not in data_txt_splits:
            raise ValueError("Unknown split arg {}".format(train))

        self._parse_func = parse_line
        lines = get_split_data(
            base_dir,
            folder_dir,
            self._train,
            self._split_mode,
            self._split_idx,
            self._split_version,
            use_task1_grasps,
            data_txt_splits,
            self._map_obj2class,
            self._parse_func,
            get_ot_pairs_taskgrasp,
            get_task1_hits)

        self._data = []
        self._pc = {}
        self._grasps = {}

        self._object_classes = class_list

        #self._num_object_classes = len(self._object_classes)

        start = time.time()
        correct_counter = 0

        all_object_instances = []

        self._object_task_pairs_dataset = []
        self._data_labels = []
        self._data_label_counter = {0: 0, 1: 0}

        for i in tqdm.trange(len(lines)):
            obj, obj_class, grasp_id, task, label = self._parse_func(lines[i])
            obj_class = self._map_obj2class[obj]
            all_object_instances.append(obj)
            self._object_task_pairs_dataset.append("{}-{}".format(obj, task))

            pc_file = os.path.join(data_dir, obj, "fused_pc_clean.npy")
            if pc_file not in self._pc:
                if not os.path.exists(pc_file):
                    raise ValueError(
                        'Unable to find processed point cloud file {}'.format(pc_file))
                pc = np.load(pc_file)
                pc_mean = pc[:, :3].mean(axis=0)
                pc[:, :3] -= pc_mean
                self._pc[pc_file] = pc

            grasp_file = os.path.join(
                data_dir, obj, "grasps", str(grasp_id), "grasp.npy")
            if grasp_file not in self._grasps:
                grasp = np.load(grasp_file)
                self._grasps[grasp_file] = grasp

            self._data.append(
                (grasp_file, pc_file, obj, obj_class, grasp_id, task, label))
            self._data_labels.append(int(label))
            if label:
                correct_counter += 1
                self._data_label_counter[1] += 1
            else:
                self._data_label_counter[0] += 1

        self._all_object_instances = list(set(all_object_instances))
        self._len = len(self._data)
        print('Loading files from {} took {}s; overall dataset size {}, proportion successful grasps {:.2f}'.format(
            data_txt_splits[self._train], time.time() - start, self._len, float(correct_counter / self._len)))

        self._data_labels = np.array(self._data_labels)

    def get_task_from_id(self, task_id):
        return self._tasks[task_id]

    def get_instance_from_id(self, instance_id):
        return self._all_object_instances[instance_id]

    @property
    def weights(self):
        N = self.__len__()
        weights = {
            0: float(N) /
            self._data_label_counter[0],
            1: float(N) /
            self._data_label_counter[1]}
        weights_sum = sum(weights.values())
        weights[0] = weights[0] / weights_sum
        weights[1] = weights[1] / weights_sum
        weights_data = np.zeros(N)
        weights_data[self._data_labels == 0] = weights[0]
        weights_data[self._data_labels == 1] = weights[1]
        return weights_data

    def __getitem__(self, idx):

        grasp_file, pc_file, obj, obj_class, grasp_id, task, label = self._data[idx]
        pc = self._pc[pc_file]
        pc = regularize_pc_point_count(
            pc, self._num_points, use_farthest_point=False)
        pc_color = pc[:, 3:]
        pc = pc[:, :3]

        grasp = self._grasps[grasp_file]
        task_id = self._tasks.index(task)
        class_id = self._object_classes.index(obj_class)
        instance_id = self._all_object_instances.index(obj)

        grasp_pc = get_gripper_control_points()
        grasp_pc = np.matmul(grasp, grasp_pc.T).T
        grasp_pc = grasp_pc[:, :3]

        latent = np.concatenate(
            [np.zeros(pc.shape[0]), np.ones(grasp_pc.shape[0])])
        latent = np.expand_dims(latent, axis=1)
        pc = np.concatenate([pc, grasp_pc], axis=0)

        pc, grasp = pc_normalize(pc, grasp, pc_scaling=self._pc_scaling)
        pc = np.concatenate([pc, latent], axis=1)

        if self._transforms is not None:
            pc = self._transforms(pc)

        label = float(label)

        return pc, pc_color, task_id, class_id, instance_id, grasp, label

    def __len__(self):
        return self._len

    def set_num_points(self, pts):
        self.num_points = min(int(1e4), pts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN training")
    parser.add_argument('--base_dir', default='', type=str)
    args = parser.parse_args()

    if args.base_dir != '':
        if not os.path.exists(args.base_dir):
            raise FileNotFoundError(
                'Provided base dir {} not found'.format(
                    args.base_dir))
    else:
        args.base_dir = os.path.join(os.path.dirname(__file__), '../../data')

    folder_dir = 'taskgrasp'
    _, _, _, name2wn = pickle.load(
        open(os.path.join(args.base_dir, folder_dir, 'misc.pkl'), 'rb'))

    train_transforms = transforms.Compose(
        [
            d_utils.PointcloudGraspToTensor(),
            d_utils.PointcloudGraspScale(),
            d_utils.PointcloudGraspRotate(axis=np.array([1.0, 0.0, 0.0])),
            d_utils.PointcloudGraspRotatePerturbation(),
            d_utils.PointcloudGraspRotate(axis=np.array([0.0, 1.0, 0.0])),
            d_utils.PointcloudGraspRotatePerturbation(),
            d_utils.PointcloudGraspRotate(axis=np.array([0.0, 0.0, 1.0])),
            d_utils.PointcloudGraspRotatePerturbation(),
            d_utils.PointcloudGraspTranslate(),
            d_utils.PointcloudGraspJitter(),
            # d_utils.PointcloudGraspRandomInputDropout(),
        ]
    )

    class_list = pickle.load(open(os.path.join(args.base_dir, 'class_list.pkl'),'rb'))
    dset = SGNTaskGrasp(
        4096,
        transforms=train_transforms,
        train=1,  # train
        base_dir=args.base_dir,
        folder_dir=folder_dir,
        normal=False,
        tasks=TASKS,
        map_obj2class=name2wn,
        class_list=class_list,
        split_mode='o',
        split_idx=0,
        split_version=0,
        pc_scaling=True,
        use_task1_grasps=True
    )

    # TODO: enable weighted sampler to have equal number of positive/negative samples
    """
    weights = dset.weights
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights, len(weights))

    dloader = torch.utils.data.DataLoader(
        dset,
        batch_size=16,
        sampler=sampler)
    """
    dloader = torch.utils.data.DataLoader(
        dset,
        batch_size=1)

    with torch.no_grad():
        for batch in dloader:
            fused_pc_xyz, pc_color, task_id, class_id, instance_id, grasp, label = batch

            grasp_pc = fused_pc_xyz[:, -7:, :]
            object_xyz = fused_pc_xyz[:, :-7, :]
            grasp_pc = grasp_pc[:, :, :3]
            object_xyz = object_xyz[:, :, :3]
            object_pc = torch.cat([object_xyz.float(), pc_color.float()], dim=-1)

            task_name = TASKS[task_id]
            print(f"visualizing for task {task_name} ({task_id.numpy()[0]}) -> label {label.numpy()[0]}")
            visualize_pc(object_pc[0], grasp_pc[0], label.numpy()[0])
