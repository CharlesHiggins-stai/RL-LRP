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
from internal_utils import filter_top_percent_pixels_over_channels, update_dictionary_patch, log_memory_usage, free_memory
from experiments import HybridCosineDistanceCrossEntopyLoss, WrapperNet

model_names = sorted(name for name in vgg.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("vgg")
                     and callable(vgg.__dict__[name]))

global args, best_prec1


def main():
    # wandb.init(project = "reverse_LRP_mnist",
    #     sync_tensorboard=True
    #     )
    extra_args = {
        'Experiment Class': 'VGG11 Hybrid Loss'
    }   
    wandb.config.update(extra_args) 
    # cheap and lazy workaround to update the config for sweeps
    update_config_for_sweeps()
    print(wandb.config)
    # set best precision to 0
    best_prec1 = 0
    # make sure we have a save dir
    if not os.path.exists(wandb.config.save_dir):
        os.makedirs(wandb.config.save_dir)
    # load the teacher model
    teacher_model = WrapperNet(get_teacher_model(wandb.config.teacher_checkpoint_path), hybrid_loss=True)
    learner_model = vgg.__dict__[wandb.config.arch]()
    learner_model.features = torch.nn.DataParallel(learner_model.features)
    learner_model = WrapperNet(learner_model, hybrid_loss=True)
    if wandb.config.cpu:
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
        batch_size=wandb.config.batch_size, shuffle=True,
        num_workers=wandb.config.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='/home/tromero_client/RL-LRP/baselines/trainVggBaselineForCIFAR10/data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=wandb.config.batch_size, shuffle=False,
        num_workers=wandb.config.workers, pin_memory=True)

    cudnn.benchmark = True

    # define loss function (criterion) and pptimizer
    criterion = HybridCosineDistanceCrossEntopyLoss(_lambda=wandb.config._lambda, mode=wandb.config.mode, step_size=wandb.config.step_size, max_lambda=wandb.config.max_lambda, min_lambda=wandb.config.min_lambda)
    if wandb.config.cpu:
        criterion = criterion.cpu()
    else:
        criterion = criterion.cuda()
        
    if wandb.config.half:
        teacher_model.half()
        learner_model.half()
        criterion.half()

    optimizer = torch.optim.SGD(learner_model.parameters(), wandb.config.lr,
                                momentum=wandb.config.momentum,
                                weight_decay=wandb.config.weight_decay)


    if wandb.config.evaluate:
            validate(val_loader, learner_model, teacher_model, criterion)
            return

    for epoch in range(wandb.config.start_epoch, wandb.config.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        if wandb.config.teacher_heatmap_mode == 'ground_truth_target':
            train_only_on_positive(train_loader, learner_model, teacher_model, criterion, optimizer, epoch)
        else:
            train(train_loader, learner_model, teacher_model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, learner_model, teacher_model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        wandb.log({
            'test/best_prec1': best_prec1
        })
        date_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        wandb.log({
            "test/best_prec1": best_prec1, 
            "epoch": epoch + 1
        })
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'state_dict': learner_model.state_dict(),
        #     'best_prec1': best_prec1,
        # }, is_best, filename=os.path.join(wandb.config.save_dir, f'checkpoint_{epoch}_{date_time}.tar'))
    # moved to save models only after the training run is done
    date_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    save_checkpoint({
            'epoch': epoch,
            'state_dict': learner_model.state_dict(),
            'best_prec1': best_prec1,
    }, is_best, filename=os.path.join(wandb.config.save_dir, f'checkpoint_{epoch}_{date_time}.tar'))
    wandb.finish()
    
def train_only_on_positive(train_loader, learner_model, teacher_model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    assert isinstance(learner_model, WrapperNet) and isinstance(teacher_model, WrapperNet), "Models must be wrapped in WrapperNet class"
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_total = AverageMeter()
    losses_cosine = AverageMeter()
    losses_cross_entropy = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    learner_model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        if wandb.config.cpu == False:
            input = input.cuda()
            target = target.cuda()
        if wandb.config.half:
            input = input.half()

        ######################################################
        #COPMUTE THE HYBIRD LOSS ON ONLY THE POSITIVE SAMPLES#
        ######################################################
        with torch.no_grad():
            classifictions = nn.functional.log_softmax(learner_model.model(input), dim=1)
            classifications = classifictions.argmax(dim=1)
            # Find the indices where classifications and labels agree
            matching_indices = (classifications == target).nonzero(as_tuple=True)[0]
            pruned_inputs = input[matching_indices]
            pruned_targets = target[matching_indices]
            if pruned_targets.shape[0]>0:
                _, target_maps = teacher_model(pruned_inputs)
        # now track gradients for forward pass  
        if pruned_targets.shape[0]>0:
            pruned_output, pruned_heatmaps = learner_model(pruned_inputs)
            target_maps = filter_top_percent_pixels_over_channels(target_maps.detach(), wandb.config.top_percent)
            loss, cosine_loss, cross_entropy_loss = criterion(pruned_heatmaps, target_maps, pruned_output, pruned_targets)
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(learner_model.parameters(), max_norm=1.0)  # Clip gradients
        ######################################################
        ##### COPMUTE THE REGULAR LOSS ON ALL SAMPLES ########
        ######################################################
        # now perform a regular backwards step using only cross entropy for all
        full_output = nn.functional.log_softmax(learner_model.model(input), dim=1)
        ce_loss = criterion.cross_entropy_loss(full_output, target)
        optimizer.zero_grad()
        ce_loss.backward()
        torch.nn.utils.clip_grad_norm_(learner_model.parameters(), max_norm=1.0)  # Clip gradients
        optimizer.step()
        update_hybrid_loss(criterion)

        output = full_output.float()
        loss = loss.float()
        cosine_loss = cosine_loss.float()
        cross_entropy_loss = cross_entropy_loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses_total.update(loss.item(), input.size(0))
        losses_cosine.update(cosine_loss.item(), input.size(0))
        losses_cross_entropy.update(cross_entropy_loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        
        # remove and reapply hooks to avoid memory leaks. 
        learner_model.remove_hooks()
        teacher_model.remove_hooks()
        learner_model.reapply_hooks()
        teacher_model.reapply_hooks()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % wandb.config.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Cosine Loss {cosine_loss.val:.4f} ({cosine_loss.avg:.4f})\t'
                  'Cross Entropy Loss {cross_entropy_loss.val:.4f} ({cross_entropy_loss.avg:.4f})\t'
                  'loss_lambda {loss_lambda:.4f}\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses_total, cosine_loss=losses_cosine, cross_entropy_loss=losses_cross_entropy, top1=top1, loss_lambda=criterion._lambda
                      ))
    wandb.log({
        "train/epoch": epoch,
        "train/loss": losses_total.avg,
        'train/loss_cosine': losses_cosine.avg,
        'train/loss_cross_entropy': losses_cross_entropy.avg,
        'train/accuracy_avg': top1.avg,
        'train/accuracy_top1': top1.val,
        'train/loss_lambda': criterion._lambda
    })
    # clean up memory
    log_memory_usage(wandb_log=False)
    free_memory()
    del batch_time, data_time, losses_total, losses_cosine, losses_cross_entropy, top1
    del output, pruned_output, pruned_heatmaps, target_maps, loss, cosine_loss, cross_entropy_loss, prec1

def train(train_loader, learner_model, teacher_model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    assert isinstance(learner_model, WrapperNet) and isinstance(teacher_model, WrapperNet), "Models must be wrapped in WrapperNet class"
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_total = AverageMeter()
    losses_cosine = AverageMeter()
    losses_cross_entropy = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    learner_model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        if wandb.config.cpu == False:
            input = input.cuda()
            target = target.cuda()
        if wandb.config.half:
            input = input.half()

        # compute output
        with torch.no_grad():
            # rather than computing the heatmaps based on the label, or simply the most activated neuron
            # we calculate the heatmap based on the selected class by the learner model, and show what the teacher model
            # would likely see as the heatmap for that class.
            if wandb.config.teacher_heatmap_mode == 'learner_label':
                output, heatmaps = learner_model(input)
                _, target_maps = teacher_model(input, output.detach().argmax(dim=1))
            elif wandb.config.teacher_heatmap_mode == 'default':
                _, target_maps = teacher_model(input)
            else:
                raise ValueError("Incorrect value for teacher_heatmap_mode")
        target_maps = filter_top_percent_pixels_over_channels(target_maps.detach(), wandb.config.top_percent)
        # now compute forward pass with grad
        if wandb.config.teacher_heatmap_mode == 'learner_label':
            output, heatmaps = learner_model(input, target)
        elif wandb.config.teacher_heatmap_mode == 'default':
            output, heatmaps = learner_model(input)
        else:
            raise ValueError("Incorrect code executed here --- something done gone fucked up")
        loss, cosine_loss, cross_entropy_loss = criterion(heatmaps, target_maps, output, target)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(learner_model.parameters(), max_norm=0.5)  # Clip gradients
        optimizer.step()
        update_hybrid_loss(criterion)

        output = output.float()
        loss = loss.float()
        cosine_loss = cosine_loss.float()
        cross_entropy_loss = cross_entropy_loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses_total.update(loss.item(), input.size(0))
        losses_cosine.update(cosine_loss.item(), input.size(0))
        losses_cross_entropy.update(cross_entropy_loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        
        # remove and reapply hooks to avoid memory leaks. 
        learner_model.remove_hooks()
        teacher_model.remove_hooks()
        learner_model.reapply_hooks()
        teacher_model.reapply_hooks()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % wandb.config.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Cosine Loss {cosine_loss.val:.4f} ({cosine_loss.avg:.4f})\t'
                  'Cross Entropy Loss {cross_entropy_loss.val:.4f} ({cross_entropy_loss.avg:.4f})\t'
                  'loss_lambda {loss_lambda:.4f}\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses_total, cosine_loss=losses_cosine, cross_entropy_loss=losses_cross_entropy, top1=top1, loss_lambda=criterion._lambda
                      ))
    wandb.log({
        "train/epoch": epoch,
        "train/loss": losses_total.avg,
        'train/loss_cosine': losses_cosine.avg,
        'train/loss_cross_entropy': losses_cross_entropy.avg,
        'train/accuracy_avg': top1.avg,
        'train/accuracy_top1': top1.val,
        'train/loss_lambda': criterion._lambda
    })
    # clean up memory
    log_memory_usage(wandb_log=False)
    free_memory()
    del batch_time, data_time, losses_total, losses_cosine, losses_cross_entropy, top1
    del output, heatmaps, target_maps, loss, cosine_loss, cross_entropy_loss, prec1

def validate(val_loader, learner_model, teacher_model, criterion, epoch):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses_total = AverageMeter()
    losses_cosine = AverageMeter()
    losses_cross_entropy = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    learner_model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if wandb.config.cpu == False:
            input = input.cuda()
            target = target.cuda()

        if wandb.config.half:
            input = input.half()

        # compute output
        with torch.no_grad():
            output, heatmaps = learner_model(input)
            _, target_maps = teacher_model(input)
            target_maps =  filter_top_percent_pixels_over_channels(target_maps.detach(), wandb.config.top_percent)
            loss, cosine_loss, cross_entropy_loss = criterion(heatmaps, target_maps, output, target)

        output = output.float()
        loss = loss.float()
        cosine_loss = cosine_loss.float()
        cross_entropy_loss = cross_entropy_loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses_total.update(loss.item(), input.size(0))
        losses_cosine.update(cosine_loss.item(), input.size(0))
        losses_cross_entropy.update(cross_entropy_loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        
        # clear hooks for clean memory management
        learner_model.remove_hooks()
        teacher_model.remove_hooks()
        learner_model.reapply_hooks()
        teacher_model.reapply_hooks()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % wandb.config.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Cosine Loss {cosine_loss.val:.4f} ({cosine_loss.avg:.4f})\t'
                  'Cross Entropy Loss {cross_entropy_loss.val:.4f} ({cross_entropy_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses_total, 
                      cosine_loss=losses_cosine, cross_entropy_loss=losses_cross_entropy,
                      top1=top1))
    wandb.log({
        "test/loss_val": losses_total.avg,
        'test/loss_cosine': losses_cosine.avg,
        'test/loss_cross_entropy': losses_cross_entropy.avg,
        'test/accuracy_top1': top1.val,
        'test/accuracy_avg': top1.avg, 
        'test/epoch': epoch
    })

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    --- overwrite any data which is present already. 
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
    lr = wandb.config.lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def update_hybrid_loss(criterion):
    if criterion.mode != None:
        criterion.step_lambda()

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

def update_config_for_sweeps():
    default_args = {
        'arch': 'vgg11',
        'workers': 0,
        'epochs': 300,
        'start_epoch': 0,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'print_freq': 20,
        'resume': '',
        'evaluate': False,
        'pretrained': False,
        'half': False,
        'cpu': False,
        'top_percent': 0.5,
        'save_dir': 'data/save_vgg11',
        'max_lambda': 0.5,
        'min_lambda': 0.0,
        'teacher_checkpoint_path': '/home/tromero_client/RL-LRP/baselines/trainVggBaselineForCIFAR10/save_vgg11/checkpoint_299.tar', 
        'teacher_heatmap_mode': 'learner_label'
    }   
    for key, value in default_args.items():
        if key not in wandb.config:
            wandb.config.update({key: value})
            print('updated wandb config to include key: ', key, ' with value: ', value)
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg11',
                        choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) +
                        ' (default: vgg19)')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
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
                        default='data/save_vgg11', type=str)
    parser.add_argument('--_lambda', type=float, default=0.1, help='balance the loss between cross entropy and cosine distance loss')
    parser.add_argument('--mode', type=str, default=None, help='mode for stepping lambda')
    parser.add_argument('--step_size', type=float, default=1e-5, help='step size for lambda')
    parser.add_argument('--max_lambda', type=float, default=0.5, help='max value for lambda')
    parser.add_argument('--min_lambda', type=float, default=0.0, help='min value for lambda')
    parser.add_argument('--teacher_checkpoint_path', type=str, help='path to teacher model checkpoint',
                        default="/home/tromero_client/RL-LRP/baselines/trainVggBaselineForCIFAR10/save_vgg11/checkpoint_299.tar")
    parser.add_argument('--teacher_heatmap_mode', type=str, help='mode for generating teacher heatmaps, options are learner_label, ground_truth_target and default', default='learner_label')
    
    args = parser.parse_args()
    # enter the main loop]
     
    wandb.init(project = "reverse_LRP_mnist",
        sync_tensorboard=True, 
        mode = 'disabled'
        )
    wandb.config.update(args)
    main()