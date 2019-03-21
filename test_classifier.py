import argparse
import time
import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import os.path as osp

from model.classifier import PointNetClassifier
import dataset
from model.stn import orthogonality_constraint
import util


def test(model, dataloader, device):
    model.eval()
    avg_time = util.AverageMeter()
    correct = 0
    total = 0
    with torch.no_grad():
        for _, (inputs, labels) in enumerate(dataloader, 0):
            start = time.time()
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)[0]
            end = time.time()
            avg_time.update(end - start)
            pred = torch.max(output.data, dim=1)[1]
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    acc = float(correct) / float(total)
    return acc, avg_time.val


def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Script for testing a \
                                                  PointNet classifier")
    parser.add_argument("dataset_name", type=str, choices=("ModelNet40",),
                        help="Name of dataset")
    parser.add_argument("snapshot", default="", type=str,
                        help="Snapshot to load weights from")
    parser.add_argument("--num_points", default=2048, type=int,
                        help="Number of points to sample from pointcloud")
    parser.add_argument("--batch_size", default=64, type=int,
                        help="Batch size for testing")
    parser.add_argument("--train_set", action="store_true",
                        help="Whether to test training set (default: False)")
    parser.add_argument("--dataset_dir", type=str, default="./",
                        help="Root directory of datasets")
    args = parser.parse_args()

    # Create model
    num_classes = 0
    if args.dataset_name == "ModelNet40":
        num_classes = 40
    model = PointNetClassifier(num_classes=num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Load weights if resuming training
    model.load_state_dict(torch.load(args.snapshot))

    # Dataset
    pin = torch.cuda.is_available()
    dset = dataset.ModelNet(dataset_path=osp.join(args.dataset_dir,
                                                  args.dataset_name),
                            training=args.train_set)
    dataloader = DataLoader(dset, batch_size=args.batch_size,
                            shuffle=True, pin_memory=pin, num_workers=4)

    # Print useful info
    print("Testing PointNet Classifier")
    print("Dataset: {}/{}".format(args.dataset_dir, args.dataset_name))
    print("Split: {}".format("train" if args.train_set else "test"))
    print("Snapshot: {}".format(args.snapshot))

    # Do testing
    start = time.time()
    accuracy, avg_time_per_batch = test(model, dataloader, device)
    elapsed = time.time() - start
    print()
    print('Accuracy: {:.2%}, elapsed time: {:.3f}s, avg. time/batch: {:.3f}s'
          .format(accuracy, elapsed, avg_time_per_batch))



if __name__ == "__main__":
    main()
