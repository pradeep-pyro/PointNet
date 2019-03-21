import numpy as np
import os
import os.path as osp
import torch
from torch.utils.data import Dataset


def _create_input_pairs(dataset_path, training):
    folders = [fol for fol in os.listdir(dataset_path)
               if osp.isdir(osp.join(dataset_path, fol))]
    label_strings = dict(zip(range(len(folders)), folders))
    data_pairs = []
    for idx, folder in enumerate(folders):
        subfolder = osp.join(dataset_path, folder, 'train' if training else 'test')
        files_in_subfolder = os.listdir(subfolder)
        data_pairs.extend([(osp.join(subfolder, fn), idx) for fn in files_in_subfolder])
    return data_pairs, label_strings


def _parse_off_vertices(filename):
    lines = None
    with open(filename, 'r') as f:
        lines = f.readlines()
    assert len(lines) > 2

    # Parse header
    lines[0] = lines[0].strip()
    if not lines[0].lower().startswith("off"):
        raise ValueError("OFF file should have a valid header")
    else:
        lines[0] = lines[0][3:]
    if len(lines[0]) > 0:
        i = 0
    else:
        i = 1
    while lines[i].startswith("#"):
        i += 1
    num_vertices = int(lines[i].split(' ')[0])

    vertices = np.empty((num_vertices, 3), dtype=np.float32)
    cnt = 0
    for line in lines[i+1:]:
        if line.startswith('#'):
            continue
        if cnt == num_vertices:
            break
        vertex = list(map(float, line.strip().split(' ')))
        vertices[cnt, 0] = vertex[0]
        vertices[cnt, 1] = vertex[1]
        vertices[cnt, 2] = vertex[2]
        cnt += 1
    return vertices


class ModelNet(Dataset):
    def __init__(self, dataset_path, num_points=2048, training=True,
                 seed=None):
        super(ModelNet, self).__init__()
        self.training = training
        self.data_pairs, self.label_strings = _create_input_pairs(dataset_path, training)
        self.num_classes = len(self.label_strings)
        self.num_points = num_points
        self.seed = seed

    def index_to_label(self, index):
        return self.label_strings.get(index, 'Unknown')
    
    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        model_path, label = self.data_pairs[idx]
        vertices = _parse_off_vertices(model_path)

        # Randomly choose num_points points from the model vertices
        if self.seed is not None:
            np.random.seed(self.seed)
        point_indices = np.random.choice(len(vertices), self.num_points,
                                         replace=True)

        vertices = vertices[point_indices, :]

        # Center the point cloud at origin
        centroid = np.mean(vertices, axis=0, keepdims=True)
        vertices -= centroid

        # Scale within unit cube
        max_dist = np.max(np.linalg.norm(vertices, axis=1))
        vertices /= max_dist

        vertices_t = torch.from_numpy(vertices.T)
        label_t = torch.tensor(label, dtype=torch.long)
        return vertices_t, label_t

