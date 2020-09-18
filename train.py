#from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import logging
#from gpu  import check_gpus

from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import os
import torch.nn.utils as utils

import numpy as np
from tqdm import tqdm
from model import DeepSpeakerModel, ResNet_coral
from eval_metrics_new import evaluate
#from DeepSpeakerDataset_dynamic import DeepSpeakerDataset_softmax, DeepSpeakerDataset_Variable, DeepSpeakerDataset_Full,DeepSpeakerDataset_Variable_all

from model_softmax import PairwiseDistance,PairwiseSimilarity

import time,random
from utils.EER import EER,cosine
from utils.load_data_BNM import load_npy_train,load_npy_train_target
import datetime
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plot
#import matplotlib.pyplot as plot
# import matplotlib as mpl
# mpl.use('TkAgg')
# import matplotlib.pyplot as plot
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Speaker Recognition')
# Model options
parser.add_argument('--dataroot', type=str, default='/nobackup/s3/projects/srlr/yaoshengyu/vox1/wav',#'/nobackup/s1/srlr/yaoshengyu/data/wav_vad',
                    help='path to dataset')
parser.add_argument('--test-pairs-path', type=str, default='/nobackup/s3/projects/srlr/yaoshengyu/DeepSpeaker-pytorch-master/data/veri_test.txt',
                    help='path to pairs file')
parser.add_argument('--log-dir', default='./data/pytorch_speaker_logs',
                    help='folder to output model checkpoints')
parser.add_argument('--resume',
                    default=None,
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=40, metavar='E',
                    help='number of epochs to train (default: 10)')
# Training options
parser.add_argument('--embedding-size', type=int, default=128, metavar='ES',
                    help='Dimensionality of the embedding')
parser.add_argument('--batch-size', type=int, default=32, metavar='BS',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='BST',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--test-input-per-file', type=int, default=1, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')
parser.add_argument('--loss-ratio', type=float, default=0.05, metavar='LOSSRATIO',
                    help='the ratio softmax loss - triplet loss (default: 2.0')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.125)')
parser.add_argument('--lr-decay', default=1e-5, type=float, metavar='LRD',
                    help='learning rate decay ratio (default: 1e-4')
parser.add_argument('--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 0.0)')
parser.add_argument('--optimizer', default='sgd', type=str,
                    metavar='OPT', help='The optimizer to use (default: Adagrad)')
parser.add_argument('--init-clip-max-norm', default=1.0, type=float,
                    metavar='CLIP', help='grad clip max norm (default: 1.0)')
# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=2, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=100, metavar='LI',
                    help='how many batches to wait before logging training status')

parser.add_argument('--mfb', action='store_true', default=True,
                    help='start from MFB file')
parser.add_argument('--makemfb', action='store_true', default=False,
                    help='need to make mfb file')

args = parser.parse_args()

# set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)


args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

if args.cuda:
    cudnn.benchmark = False
torch.backends.cudnn.enabled = True

LOG_DIR = args.log_dir
os.system('chmod 777 %s/*'%LOG_DIR)

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
	
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
l2_dist = PairwiseDistance(2)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n=1):
        self.val   = val
        self.sum   += val * n
        self.count += n
        self.avg   = self.sum / self.count

def myaccuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred    = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res



def main():
    model = ResNet_coral(embedding_size=args.embedding_size,num_classes=1211,dropout_p=0.0,
                    batch_norm=True)
    model.cuda()
    model = torch.nn.DataParallel(model)
    print('------------------ initial finished ----------------------\n')
    print('\nparsed options:\n{}\n'.format(vars(args)))
    print(LOG_DIR)
    optimizer = create_optimizer(model, args.lr)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            optimizer.param_groups[0]['lr']=args.lr
            loss_mean=np.load(LOG_DIR+'/loss_%s.npy'%(args.start_epoch-1)).tolist()
            tmp=[]
            for line in open(LOG_DIR+'/loss.txt'):
                if int(line.split(':')[1].split()[0])<args.start_epoch:
                    tmp.append(line)
            fp=open(LOG_DIR+'/loss.txt', 'w')
            for i in range(len(tmp)):
                fp.write(tmp[i])
            fp.close()
        else:
            loss_mean=[]
            print('=> no checkpoint found at {}'.format(args.resume))
    #load test data
    print("Read valid data...")
    valid = {}
    utt_list = []
    for line in open('trial-core-core'):
        line=line.strip().split(' ')
        label = int(line[0]) 
        #line[1] = '-'.join(line[1].split('/')).split('.')[0]
        #line[2] = '-'.join(line[2].split('/')).split('.')[0]
        if line[1] not in utt_list:
            utt_list.append(line[1])
        if line[2] not in utt_list:
            utt_list.append(line[2])
    print('...................... read valid data .........................\n')
    
    for key in utt_list:
        mat = np.load('%s/%s.npy' % ('/home/chenzhigao/SITW/sitw_eval_fbank', key))
        mat = np.reshape(mat, [1, 1, mat.shape[0], mat.shape[1]])
        valid[key] = torch.from_numpy(mat).float()
    print("Read valid data done.")
    print(torch.cuda.device_count())
    #train
    start = args.start_epoch
    end = start + args.epochs
    label_train={}
    label_domain_s={}
    label_domain_t={}
    utt_train=[]
    utt_train_target=[]
    print('...................... read train data .........................\n')
    for line in open('/home/xiaorunqiu/forReaearch/vox/data_fbank/train_npy.list'):
        line=line.strip().split(' ')
        label_train[line[0]] = int(line[1])
        label_domain_s[line[0]] = torch.Tensor([1,0])
        utt_train.append(line[0])

    for line in open('sitw_dev_npy_sup.list'):
        line=line.strip().split(' ')
        label_domain_t[line[0]] = torch.Tensor([0,1])
        utt_train_target.append(line[0])
    
    print('...................... start train .........................\n')
    for epoch in range(start, end):
        #Valid(valid, model, epoch)
        print('epoch = %d\n',epoch)
        train(utt_train, label_train,label_domain_s, utt_train_target,label_domain_t, model, optimizer, epoch, loss_mean)
        Valid(valid, model, epoch)


def train(utt_train, label_train,label_domain_s, utt_train_target,label_domain_t, model, optimizer, epoch, loss_mean):
    # switch to train mode
    model.train()
    batch_size = args.batch_size
    top1       = AverageMeter() 
    top5       = AverageMeter()

    random.shuffle(utt_train)
    random.shuffle(utt_train_target)
    epoch_itr = int(len(utt_train) / batch_size)
    loss_tmp = 0.0
    CEloss_tmp=0.0
    dom_tmp=0.0
    coral_tmp=0.0
    for itr in range(epoch_itr):
        #t1=time.time()
        chunk_ran=random.randint(200,600)#for concatenate,keep chunk equal
        input_batch, label_batch, label_domain_batch= load_npy_train(utt_train[itr*batch_size:(itr+1)*batch_size], 
                                                    label_train, 
                                                    label_domain_s,
                                                    '/home/xiaorunqiu/forReaearch/vox/data_fbank/train/', 
                                                    chunk_ran,
                                                    do_aug=False)
        input_batch_target, label_domain_batch_target = load_npy_train_target(utt_train_target[itr*batch_size:(itr+1)*batch_size],label_domain_t,'/home/chenzhigao/SITW/sitw_dev_fbank',chunk_ran,do_aug=False)
        
        input_batch_mix=np.concatenate((input_batch,input_batch_target),axis=0)
        label_domain_mix=np.concatenate((label_domain_batch,label_domain_batch_target),axis=0)
        state=np.random.get_state()
        np.random.shuffle(input_batch_mix)
        np.random.set_state(state)
        np.random.shuffle(label_domain_mix)
        
        
        data = torch.from_numpy(input_batch).float()
        data = data.cuda()
        data_var= Variable(data)
        #print("data_var:",data_var.size())
 
        label = torch.from_numpy(label_batch).long()
        label = label.cuda() 
        label_var = Variable(label)
        
        data_mix = torch.from_numpy(input_batch_mix).float()
        data_mix = data_mix.cuda()
        data_mix_var = Variable(data_mix)
        
        label_domain = torch.from_numpy(label_domain_mix).float()
        label_domain = label_domain.cuda()
        label_domain_var = Variable(label_domain)
        
#        label_domain_s = torch.from_numpy(label_domain_batch).long()
#        label_domain_s = label_domain_s.cuda()
#        label_domain_s_var = Variable(label_domain_s)
#        
#        label_domain_t = torch.from_numpy(label_domain_batch_target).long()
#        label_domain_t = label_domain_t.cuda()
#        label_domain_t_var = Variable(label_domain_t)

        data_target = torch.from_numpy(input_batch_target).float()
        data_target = data_target.cuda()
        data_var_target= Variable(data_target)
        #print("data_Var_target:",data_var_target.size())

        out_feature, out_cls, out_dom= model(data_var)
        out_feature_target, out_cls_target, out_dom_target=model(data_var_target)
        out_feature_mix, out_cls_mix, out_dom_mix = model(data_mix_var)
        
        dim=out_feature.size(1)
        ns=out_feature.size(0)
        nt=out_feature_target.size(0)
        tmp=torch.ones(1,ns)
        tmp=tmp.cuda()
        tmp_s=tmp @ out_feature
        cs=(out_feature.t() @ out_feature-(tmp_s.t()@tmp_s)/ns)/(ns-1)
        tmp_t=tmp@out_feature_target
        ct=(out_feature_target.t() @ out_feature_target-(tmp_t.t()@tmp_t)/nt)/(nt-1)
        coral_loss=(cs-ct).pow(2).sum().sqrt()
        coral_loss=coral_loss/(4*dim*dim)

        criterion = nn.CrossEntropyLoss()
        cross_entropy_loss = criterion(out_cls, label_var)

        criterion_dom = nn.BCELoss()
        sig = nn.Sigmoid()
        domain_loss = criterion_dom(sig(out_dom_mix),label_domain_var)

        prec1, prec5 = myaccuracy(out_cls.data, label, topk=(1, 5))
        top1.update(prec1.item(), data.size(0))
        top5.update(prec5.item(), data.size(0))
        
        loss = cross_entropy_loss-0.1*domain_loss+100*coral_loss
        #loss = cross_entropy_loss
#        loss = cross_entropy_loss+100*coral_loss
        optimizer.zero_grad()
        loss.backward()
        loss_tmp += loss.item()
        CEloss_tmp+=cross_entropy_loss.item()
        dom_tmp+=domain_loss.item()
        coral_tmp+=coral_loss.item()
        #if args.init_clip_max_norm is not None:
        #    utils.clip_grad_norm(model.parameters(),
        #            max_norm=args.init_clip_max_norm)
        utils.clip_grad_value_(model.parameters(), args.init_clip_max_norm)
        optimizer.step()
        #print(time.time() - t2, "seconds process time run network")

        if itr % args.log_interval == 0 and itr!=0:
            loss_tmp /= args.log_interval
            CEloss_tmp /= args.log_interval
            dom_tmp /= args.log_interval
            coral_tmp /= args.log_interval
            #print('Train Epoch: {:3d} [{:8d}/{:8d} ({:3.0f}%)]\tLoss: {:.6f}\tacc1: {:.6f}\tlr:{:.4f}'.format(
            #    epoch, itr, epoch_itr,
            #    100. * itr / epoch_itr,loss_mean,
            #    top1.avg, optimizer.param_groups[0]['lr']))
            fp = open(LOG_DIR+'/loss.txt','a')
            #fp.write('epoch:{:3d} itr:{:5d} Loss:{:.6f}'.format(epoch, itr, loss_tmp))
            fp.write('Train Epoch: {:3d} [{:8d}/{:8d} ({:3.0f}%)]\tLoss: {:.6f}\tCELoss: {:.6f}\tDomLoss: {:.6f}\tcoralLoss: {:.6f}\tacc1: {:.6f}\tlr:{:.4f}\n'.format(
                epoch, itr, epoch_itr,
                100. * itr / epoch_itr,loss_tmp,CEloss_tmp,dom_tmp,coral_tmp,
                top1.avg, optimizer.param_groups[0]['lr']))
            fp.close()
            loss_mean.append(loss_tmp)
            loss_tmp = 0.0
            CEloss_tmp=0.0
            dom_tmp=0.0
            coral_tmp=0.0
#            #plot
#            plot.figure()
#            plot.plot(range(0,args.log_interval*len(loss_mean),args.log_interval), loss_mean)
#            plot.vlines(range(0, args.log_interval*len(loss_mean), int(epoch_itr/args.log_interval)*args.log_interval), 0, 8)
#            plot.title("loss")
#            plot.xlabel("itr")
#            plot.ylabel("loss")
#            plot.savefig(LOG_DIR+'/loss.png', format='png', dpi=119)
#            #plot
#            plot.figure()
            if epoch > 10: 
                tmp=loss_mean[(epoch-10)*int(epoch_itr/args.log_interval):-1]
            else:
                tmp = loss_mean
#            plot.plot(range(0,args.log_interval*len(tmp), args.log_interval), tmp)
#            plot.vlines(range(0, args.log_interval*len(tmp), int(epoch_itr/args.log_interval)*args.log_interval), 0, max(tmp))
#            plot.title("loss")
#            plot.xlabel("itr")
#            plot.ylabel("loss")
#            plot.savefig(LOG_DIR+'/loss1.png', format='png', dpi=119)
            
            top1 = AverageMeter() 
            top5 = AverageMeter()
    # do checkpointing
    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()},
               '{}/checkpoint_{}.pth'.format(LOG_DIR, epoch))
    np.save(LOG_DIR+'loss_%s'%epoch, np.array(loss_mean))

def test(test_loader, model, epoch):
    # switch to evaluate mode
    model.eval()

    labels, distances = [], []

    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:
        #current_sample = data_a.size(0)
        #data_a = data_a.resize_(args.test_input_per_file *current_sample, 1, data_a.size(2), data_a.size(3))
        #data_p = data_p.resize_(args.test_input_per_file *current_sample, 1, data_a.size(2), data_a.size(3))
        if args.cuda:
            data_a, data_p = data_a.cuda(), data_p.cuda()
        data_a, data_p, label = Variable(data_a, volatile=True), \
                                Variable(data_p, volatile=True), Variable(label)

        # compute output
        out_a, out_p = model(data_a), model(data_p)
        dists = l2_dist.forward_l2(out_a,out_p)# euclidean distance
        #dists = l2_dist.forward(out_a,out_p) #cosine similarity
        dists = dists.data.cpu().numpy()
        #dists = dists.reshape(current_sample, 1).mean(axis=1) #args.test_input_per_file).mean(axis=1)
        distances.append(dists)
        labels.append(label.data.cpu().numpy())

        if batch_idx % args.log_interval == 0:
            pbar.set_description('Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * len(data_a), len(test_loader.dataset),
                100. * batch_idx / len(test_loader)))

    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])

    tpr, fpr, accuracy, val,  far, eer= evaluate(distances,labels)
    fp = open(LOG_DIR+'/log.txt','a')
    fp.write('epoch: {}. Test set: Accuracy: {:.8f}, EER:{:.8f}\n'.format(epoch, np.mean(accuracy),np.mean(eer)))
    fp.close()
    print('\33[91mTest set: Accuracy: {:.8f}, EER:{:.8f}\n\33[0m'.format(np.mean(accuracy),np.mean(eer)))
def Valid(valid, model, epoch):
    model.eval()    
    uttrance_feature={}
    for key in valid: 
        with torch.no_grad():
            if args.cuda:
                data = valid[key].cuda()
            data = Variable(data) 
            uttrance_feature[key] = model(data)[0][0].cpu()
            #output = model(data)
            
    score_gen=[]
    score_spo=[]
    for line in open('trial-core-core'):
        line=line.strip().split(' ')
        label = int(line[0]) 
        #line[1] = '-'.join(line[1].split('/')).split('.')[0]
        #line[2] = '-'.join(line[2].split('/')).split('.')[0]
        score = torch.nn.functional.cosine_similarity(uttrance_feature[line[1]], uttrance_feature[line[2]], dim=0).item()
        if label==1:
            score_gen.append(score)
        else:
            score_spo.append(score)
    eer, _ = EER(score_gen, score_spo)
    #print("Epoch:%s EER:%.2f Positive:%s Negative:%s"%(epoch, eer, len(score_gen), len(score_spo)))
    fp = open(LOG_DIR+'/log.txt','a')
    fp.write('epoch: {}.\tEER:{:.3f}\n'.format(epoch, eer))
    fp.close()
    
def create_optimizer(model, new_lr):
    # setup optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.9, dampening=0.9,
                              weight_decay=args.wd)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr,
                               weight_decay=args.wd)
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(),
                                  lr=new_lr,
                                  lr_decay=args.lr_decay,
                                  weight_decay=args.wd)
    #optimizer = nn.DataParallel(optimizer, device_ids=device_ids)
    #torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=40, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    
    return optimizer

if __name__ == '__main__':
    main()
