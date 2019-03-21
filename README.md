# PointNet in PyTorch

This is a PyTorch implementation of PointNet by Qi et al. published in CVPR 2017.

Summary of modules:

- `PointNetFeatures` (pointnet/model/pointnet.py): extracts point cloud features
- `PointNetClassifier` (pointnet/model/classifier.py): model for classification that builds upon `PointNetFeatures`
- `PointSpatialTransformer` (pointnet/model/stn.py): spatial transformer network for point clouds

## Usage

```python
python train_classifier.py ModelNet40 --dataset_dir /path/to/datasets --snapshot_dir /path/to/snapshots --snapshot_every 10 --with_validation --epochs 100
```
converges to ~85% accuracy after 100 epochs.

```python
python test_classifier.py ModelNet40 /path/to/snapshots/best --dataset_dir /path/to/datasets
```
