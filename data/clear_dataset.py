from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip,Resize
from torch.utils.data import Dataset,Subset
import torchvision.transforms as transforms
import os
import numpy as np
import random
from PIL import Image
import torch
import pdb
# from avalanche.benchmarks  import benchmark_with_validation_stream
from .parse_data_path import *
'''
timestamp_index stand for, for each timestamp, the index of instance in the txt file, since each subset represent
one timestamp data, thus we need to have the index of data of each timestamp
timestamp_index[0] == the set of index of data belong to bucket 1
'''
def get_instance_time(args,idx,all_timestamp_index):
    for index,list in enumerate(all_timestamp_index):
        if(idx in list):
            return index
    assert False, "couldn't find timestamp info for data with index {}".format(idx)

def get_feature_extract_loader(args):
    dataset=CLEARDataset(args,data_txt_path='../{}/data_cache/data_all_path.txt'.format(args.split),stage='all')
    all_timestamp_index=dataset.get_timestamp_index()
    return dataset,all_timestamp_index


def prepare_clear_samples(args, data_txt_path):
    if(os.path.isfile(data_txt_path)==False):
        print('loading data_list from folder')
        parse_data_meta_path(args)
    else:
        print('loaded exist data_list')
    n_env = args.n_env
    samples=[]
    targets=[]
    all_up_ts = []
    all_tk_ts = []
    timestamp_index=[[] for i in range(n_env)]
    index=0
    with open(data_txt_path,'r') as file:
        title=file.readline()
        while (True):
            line=file.readline()
            if(line==''):
                break
            #'/data3/zhiqiul/yfcc_dynamic_10/dynamic_300/images/bucket_6/racing/6111026970.jpg 8 6 1657562 18778266\n'
            line_list=line.split()
            if int(line_list[1]) == 0:
                continue
            targets.append(int(line_list[1]) - 1)
            timestamp_index[int(line_list[2])-1].append(index)
            samples.append(line_list)
            up_ts, tk_ts = int(line_list[3]), int(line_list[4])
            all_up_ts.append(up_ts)
            all_tk_ts.append(tk_ts)
            index=index+1
            if(index%10000==0):
                print('finished processing data {}'.format(index))
            
    # self.targets=targets
    samples=samples
    timestamp_index=timestamp_index
    mean_up, std_up, max_up, min_up = np.mean(all_up_ts), np.std(all_up_ts), np.max(all_up_ts), np.min(all_up_ts)
    max_up = 1356998400 # 2013-01-01:00:00:00
    mean_tk, std_tk, max_tk, min_tk = np.mean(all_tk_ts), np.std(all_tk_ts), np.max(all_tk_ts), max(1072886400, np.min(all_tk_ts))
    
    all_stats = [mean_up, std_up, max_up, min_up, mean_tk, std_tk, max_tk, min_tk]
    all_samples = [[] for _ in range(10)]
    all_targets = [[] for _ in range(10)]
    for sample_id in range(len(samples)):
        sample = samples[sample_id]
        tgt = targets[sample_id]
        bck_id = int(sample[2]) - 1
        # print(bck_id)
        all_samples[bck_id].append(sample)
        all_targets[bck_id].append(tgt)
    if args.split_num > 0:
        return all_samples, all_targets, all_stats, all_up_ts
    else:
        return all_samples, all_targets, all_stats

class CLEARDatasetSWA(Dataset):
    def __init__(self, args, all_stats, all_samples, all_targets, bucket_id = 0, transform = None, only_img = True, split_pos = None):
        assert transform is None
        self.transform = transform
        self.only_img = only_img
        self.args=args
        # print(self.args)
        # input("printing dataset config right?")
        self.n_classes = args.num_classes
        self.n_experiences = args.timestamp
        assert self.n_experiences == len(all_samples)
        if split_pos is not None:
            self.samples = []
            self.targets = []
            for idb in range(len(all_samples)):
                for ids in range(len(all_samples[idb])):
                    # pdb.set_trace()
                    if int(all_samples[idb][ids][3]) >= split_pos[0] and int(all_samples[idb][ids][3]) <=split_pos[1]:
                        self.samples.append(all_samples[idb][ids])
                        self.targets.append(all_targets[idb][ids])
            self.targets=torch.from_numpy(np.array(self.targets))
        else:
            self.samples = all_samples[bucket_id]
            self.targets=torch.from_numpy(np.array(all_targets[bucket_id]))
        self.mean_up, self.std_up, self.max_up, self.min_up, self.mean_tk, self.std_tk, self.max_tk, self.min_tk = all_stats
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,index):
        
        sample, label = Image.open(self.samples[index][0]),int(self.samples[index][1]) - 1
        up_ts, tk_ts = int(self.samples[index][3]), int(self.samples[index][4])
        bucket_id = int(self.samples[index][2])
        # print(bucket_id)
        array=np.array(sample)
        # some image may have 4 channel (alpha)
        if(array.shape[-1]==4):
            array=array[:,:,:3]
        elif(array.shape[-1]==1):
            array=np.concatenate((array, array, array), axis=-1)
        elif(len(array.shape)==2):
            array=np.stack([array,array,array],axis=-1)
        # import pdb;pdb.set_trace()
        # array=np.ones(array.shape,dtype='uint8')*int(get_instance_time(self.args,index,self.timestamp_index)) # for debug
        
        sample=Image.fromarray(array)
        if self.transform is not None:
            # print("using the data augmentation")
            sample = self.transform(sample)
        if self.args.ts_norm == 'minmax':
            # print("using min max normalization!!!!")
            norm_up_ts = (up_ts - self.min_up) / (self.max_up - self.min_up + 1e-8)
            # print("using the minmax to process the timestamp", self.max_up, self.min_up, up_ts, norm_up_ts, "bucket ID:", bucket_id)
            # input("haha")
            norm_tk_ts = (tk_ts - self.min_tk) / (self.max_tk - self.min_tk + 1e-8)
            # print(tk_ts, self.min_tk, self.max_tk, norm_tk_ts)
            # input("haha")
            
        else:
            norm_up_ts = (up_ts - self.mean_up) / (self.std_up + 1e-8)
            norm_tk_ts = (tk_ts - self.mean_tk) / (self.std_tk + 1e-8)
            # print("using the standard to process the timestamp", self.mean_up, self.std_up, up_ts, norm_up_ts, "bucket ID:", bucket_id)
        if self.only_img:
            return sample, label
        else:
            return sample, label, norm_up_ts, norm_tk_ts, bucket_id, self.samples[index][0]

class CLEARSubset(Dataset):
    def __init__(self, dataset, indices, targets,bucket):
        self.dataset = torch.utils.data.Subset(dataset, indices)
        self.indices=indices
        self.targets = targets.numpy() # need to be in numpy(thus set of targets have only 10 elem,rather than many with tensor)
        self.bucket=bucket
    def get_indice(self):
        return self.indices
    def get_bucket(self):
        return self.bucket
    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        target = self.targets[idx]
        print(target, self.dataset[idx][1])
        assert int(self.dataset[idx][1])==target
        return (image, target)
    def __len__(self):
        return len(self.targets)

class CLEARSubsetCIDA(Dataset):
    def __init__(self, dataset, indices, targets,bucket):
        self.dataset = torch.utils.data.Subset(dataset, indices)
        self.indices=indices
        self.targets = targets.numpy() # need to be in numpy(thus set of targets have only 10 elem,rather than many with tensor)
        self.bucket=bucket
    def get_indice(self):
        return self.indices
    def get_bucket(self):
        return self.bucket
    def __getitem__(self, idx):
        sample, label, norm_up_ts, norm_tk_ts, bucket_id = self.dataset[idx]
        image = sample
        target = self.targets[idx]
        out_tuple = (sample, label, norm_up_ts, norm_tk_ts, bucket_id)
        # print(target, self.dataset[idx][1])
        assert int(label)==target
        return (image, target, norm_up_ts, norm_tk_ts, bucket_id)
    def __len__(self):
        return len(self.targets)

def get_transforms(args):
    # Note that this is not exactly imagenet transform/moco transform for val set
    # Because we resize to 224 instead of 256
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        # transforms.RandomResizedCrop(224),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    if(args.pretrain_feature!='None'):
        train_transform,test_transform=None,None
    return train_transform, test_transform

def get_mnist_transforms(args):
    # Note that this is not exactly imagenet transform/moco transform for val set
    # Because we resize to 224 instead of 256
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        # transforms.RandomResizedCrop(224),
        transforms.RandomResizedCrop(28),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.Resize(28),
        transforms.CenterCrop(28),
        transforms.ToTensor(),
        normalize,
    ])