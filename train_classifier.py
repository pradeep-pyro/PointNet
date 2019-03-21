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


BEST_ACCURACY = 0.0


def train(model, dataloader, optimizer, epoch, device, print_freq=10):
    model.train()
    avg_loss = util.AverageMeter()
    avg_time = util.AverageMeter()
    for i, (inputs, labels) in enumerate(dataloader, 0):
        start = time.time()
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        output, trans_inp, trans_feat = model(inputs)
        loss = F.cross_entropy(output, labels)
        loss += 0.001 * orthogonality_constraint(trans_inp)
        loss += 0.001 * orthogonality_constraint(trans_feat)
        loss.backward()
        optimizer.step()
        end = time.time()
        avg_loss.update(loss.item())
        avg_time.update(end - start)
        if i > 0 and i % print_freq == 0:
            print('Train Epoch {:3} [{:3.0f}% of {}]: Loss: {:6.3f}'
                  .format(epoch, (i + 1) / len(dataloader) * 100.0,
                          len(dataloader.dataset), loss.item()))
    return avg_loss.val, avg_time.val


def validate(model, dataloader, epoch, device):
    model.eval()
    avg_loss = util.AverageMeter()
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
            loss = F.cross_entropy(output, labels)
            avg_loss.update(loss.item())
            pred = torch.max(output.data, dim=1)[1]
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    acc = float(correct) / float(total)
    print('Test Epoch {:3}: Avg. loss: {:6.3f}, Accuracy: {:.2%}, Avg. Time/batch: {:5.3f}s'
          .format(epoch, avg_loss.val, acc, avg_time.val))
    return avg_loss.val, avg_time.val, acc


def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Script for training a \
                                                  PointNet classifier")
    parser.add_argument("dataset_name", type=str, choices=("ModelNet40",),
                        help="Name of dataset")
    parser.add_argument("--num_points", default=2048, type=int,
                        help="Number of points to sample from pointcloud")
    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="Batch size for training")
    parser.add_argument("--val_batch_size", default=64, type=int,
                        help="Batch size for validation")
    parser.add_argument("--epochs", default=100, type=int,
                        help="Number of epochs to train for")
    parser.add_argument("--resume", default="", type=str,
                        help="Resume training from snapshot")
    parser.add_argument("--dataset_dir", type=str, default="./",
                        help="Root directory of datasets")
    parser.add_argument("--snapshot_dir", default=".", type=str,
                        help="Path to snapshot directory")
    parser.add_argument("--snapshot_every", default=10, type=int,
                        help="Snapshot is saved after every X epochs")
    parser.add_argument("--with_validation", action="store_true",
                        help="Whether to perform validation after each epoch")
    args = parser.parse_args()

    # Create model
    num_classes = 0
    if args.dataset_name == "ModelNet40":
        num_classes = 40
    model = PointNetClassifier(num_classes=num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Load weights if resuming training
    if args.resume != "":
        model.load_state_dict(torch.load(args.resume))

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4,
                           weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


    # Dataset
    pin = torch.cuda.is_available()
    training_set = dataset.ModelNet(dataset_path=osp.join(args.dataset_dir,
                                                          args.dataset_name),
                                    training=True)
    val_set = dataset.ModelNet(dataset_path=osp.join(args.dataset_dir,
                                                     args.dataset_name),
                               training=False)
    train_loader = DataLoader(training_set, batch_size=args.train_batch_size,
                              shuffle=True, pin_memory=pin, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size,
                            shuffle=True, pin_memory=pin, num_workers=4)

    # Make directory for saving snapshots
    snapshot_dir = args.snapshot_dir
    if not osp.isdir(snapshot_dir):
        os.makedirs(snapshot_dir)

    # Print useful info
    print("Training PointNet Classifier")
    print("Dataset: {}/{}".format(args.dataset_dir, args.dataset_name))
    print("Snapshot path: {}".format(args.snapshot_dir))
    print("Validation after each epoch: ", str(args.with_validation))
    print("Save snapshot every: {} epoch".format(args.snapshot_every))
    print("Batch size: train: {}, val: {}".format(args.train_batch_size,
                                                  args.val_batch_size))
    print("Point samples: {}".format(args.num_points))

    print()
    print("Start time: ", time.asctime()) 

    # Do training/validation    
    for epoch in range(1, args.epochs + 1):
        scheduler.step(epoch)
        train(model, train_loader, optimizer, epoch, device)
        if epoch % args.snapshot_every == 0 or epoch == args.epochs:
            torch.save(model.state_dict(),
                       osp.join(snapshot_dir, "epoch_%d" % epoch))
        if args.with_validation:
            _, _, acc = validate(model, val_loader, epoch, device)
            global BEST_ACCURACY
            if acc > BEST_ACCURACY:
                BEST_ACCURACY = acc
                torch.save(model.state_dict(), osp.join(snapshot_dir, "best"))

    print("End time: ", time.asctime())


if __name__ == "__main__":
    main()
