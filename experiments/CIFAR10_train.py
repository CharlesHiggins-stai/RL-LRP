import sys
sys.path.append('/home/tromero_client/RL-LRP')
import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import baselines.trainVggBaselineForCIFAR10.vgg as vgg
import wandb
from internal_utils import filter_top_percent_pixels_over_channels, update_dictionary_patch 
from experiments import HybridCosineDistanceCrossEntopyLoss, WrapperNet

model_names = sorted(name for name in vgg.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("vgg")
                     and callable(vgg.__dict__[name]))




def main(args):
    # set best precision to 0
    best_prec1 = 0
    # make sure we have a save dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # load the teacher model
    teacher_model = WrapperNet(get_teacher_model(args.teacher_checkpoint_path), hybrid_loss=True)
    learner_model = vgg.__dict__[args.arch]()
    learner_model.features = torch.nn.DataParallel(learner_model.features)
    learner_model = WrapperNet(learner_model, hybrid_loss=True)
    if args.cpu:
        learner_model.cpu()
        teacher_model.cpu
    else:
        learner_model.cuda()
        teacher_model.cuda()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='/home/tromero_client/RL-LRP/baselines/trainVggBaselineForCIFAR10/data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='/home/tromero_client/RL-LRP/baselines/trainVggBaselineForCIFAR10/data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    cudnn.benchmark = True

    # define loss function (criterion) and pptimizer
    criterion = HybridCosineDistanceCrossEntopyLoss(_lambda=args._lambda)
    if args.cpu:
        criterion = criterion.cpu()
    else:
        criterion = criterion.cuda()
        
    if args.half:
        teacher_model.half()
        learner_model.half()
        criterion.half()

    optimizer = torch.optim.SGD(learner_model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)


    if args.evaluate:
            validate(val_loader, learner_model, teacher_model, criterion)
            return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, learner_model, teacher_model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, learner_model, teacher_model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': learner_model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_dir, 'checkpoint_{}.tar'.format(epoch)))

def train(train_loader, learner_model, teacher_model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    assert isinstance(learner_model, WrapperNet) and isinstance(teacher_model, WrapperNet), "Models must be wrapped in WrapperNet class"
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    learner_model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        if args.cpu == False:
            input = input.cuda()
            target = target.cuda()
        if args.half:
            input = input.half()

        # compute output
        output, heatmaps = learner_model(input)
        _, target_maps = teacher_model(input)
        target_maps = filter_top_percent_pixels_over_channels(target_maps, args.top_percent)
        loss = criterion(heatmaps, target_maps, output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
            wandb.log({
                "train/epoch": epoch,
                "train/loss": losses.avg,
                'train/accuracy_avg': top1.avg,
                'train/accuracy_top1': top1.val
            })


def validate(val_loader, learner_model, teacher_model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    learner_model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if args.cpu == False:
            input = input.cuda()
            target = target.cuda()

        if args.half:
            input = input.half()

        # compute output
        with torch.no_grad():
            output, heatmaps = learner_model(input)
            _, target_maps = teacher_model(input)
            target_maps =  filter_top_percent_pixels_over_channels(target_maps, args.top_percent)
            loss = criterion(heatmaps, target_maps, output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))
            wandb.log({
                "test/loss_val": losses.avg,
                'test/accuracy_top1': top1.val,
                'test/accuracy_avg': top1.avg
            })

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = args.lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



def get_teacher_model(teacher_checkpoint_path):
    checkpoint = torch.load(teacher_checkpoint_path)
    # assume teacher model is vgg11 for now
    teacher = vgg.vgg11()
    checkpoint = update_dictionary_patch(checkpoint)
    teacher.load_state_dict(checkpoint['new_state_dict'])
    return teacher

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg19',
                        choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) +
                        ' (default: vgg19)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--print-freq', '-p', default=20, type=int,
                        metavar='N', help='print frequency (default: 20)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--half', dest='half', action='store_true',
                        help='use half-precision(16-bit) ')
    parser.add_argument('--cpu', dest='cpu', action='store_true',
                        help='use cpu')
    parser.add_argument('--top_percent', type=float, default=0.75, 
                        help='Top x percent of pixels to retain')
    parser.add_argument('--save-dir', dest='save_dir',
                        help='The directory used to save the trained models',
                        default='save_temp', type=str)
    parser.add_argument('--_lambda', type=float, default=0.1, help='balance the loss between cross entropy and cosine distance loss')
    parser.add_argument('--teacher_checkpoint_path', type=str, help='path to teacher model checkpoint',
                        default="/home/tromero_client/RL-LRP/baselines/trainVggBaselineForCIFAR10/save_vgg11/checkpoint_299.tar")
    
    global args, best_prec1
    args = parser.parse_args()
    
    wandb.init(project = "reverse_LRP_mnist",
        sync_tensorboard=True,
        mode = "disabled"
        )
    wandb.config.update(args)
    extra_args = {
        'Experiment Class': 'VGG11 Hybrid Loss'
    }
    wandb.config.update(extra_args)
    # enter the main loop
    main(args)