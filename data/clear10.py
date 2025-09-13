import os
import os.path as osp
import pdb
import json
import pickle
import numpy as np
import random
# import numpy as np
import torch
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from wilds import get_dataset
from dateutil import parser
from pathlib import Path

import pdb
ID_HELD_OUT = 0.1

def get_ts(image_ls, meta_data):
    res_ls = []
    for image_path in image_ls:
        image_id = image_path.split('/')[-1].split('.')[0]
        upload_ts = meta_data[image_id]["DATE_UPLOADED"]
        taken_ts = str(int(parser.isoparse(meta_data[image_id]["DATE_TAKEN"]).timestamp()))
        # print(taken_ts, meta_data[image_id]["DATE_TAKEN"])
        res_ls.append((image_path, upload_ts, taken_ts))
    return res_ls

def list_all_files_meta(args,rootdir):
    print("listing all files in rootdir:", rootdir)
    # input()
    train_list,test_list,all_list = [],[],[]
    # get the meta data path
    meta_data_path = rootdir[:-14] + 'labeled_metadata'
    bucket_list = os.listdir(rootdir)
    # bucket_list=list(filter(lambda a: 'bucket_' in a,bucket_list))
    if('0' in bucket_list):
        bucket_list.remove('0') # skip bucket 0, since it's for pretrain feature
    classes_list=  os.listdir(osp.join(rootdir,bucket_list[0]))
    print(classes_list)
    # exit()
    # if('clear25d' in args.split and 'BACKGROUND' in classes_list):
    if('BACKGROUND' in classes_list):
        classes_list.remove('BACKGROUND') # skip bucket 0, since it's for pretrain feature
    print(classes_list)
    # exit()
    for bucket in bucket_list:
        for classes in classes_list:
            image_list=os.listdir(osp.join(rootdir,bucket,classes))
            image_list=list(map(lambda a: osp.join(osp.join(rootdir,bucket,classes,a)), image_list))
            image_list=image_list[:args.num_instance_each_class] # if background class have more image, we use only part of it
            # print("image_list", image_list)
            meta_file_path = os.path.join(meta_data_path, bucket, classes+'.json')
            with open(meta_file_path) as f:
                meta_data = json.load(f)
            try:
                assert len(image_list)==args.num_instance_each_class
            except:
                # import pdb;pdb.set_trace()
                print("the bucket", str(bucket), 'class', str(classes), 'has', len(image_list), 'not', args.num_instance_each_class)
                # exit()
            train_subset,test_subset=train_test_split(image_list,test_size=ID_HELD_OUT, random_state=args.random_seed)
            train_list.extend(get_ts(train_subset, meta_data))
            test_list.extend(get_ts(test_subset, meta_data))
            all_list.extend(get_ts(image_list, meta_data))

    import random
    random.seed(args.random_seed)
    random.shuffle(train_list)
    random.shuffle(test_list)
    random.shuffle(all_list)
    return train_list,test_list,all_list

def prepare_clear_samples(args):
    if(os.path.isfile(args.data_txt_path)==False):
        print('loading data_list from folder')
        parse_data_meta_path(args)
    else:
        print('loaded exist data_list')
    n_env = args.timestamp
    samples=[]
    targets=[]
    all_up_ts = []
    all_tk_ts = []
    timestamp_index=[[] for i in range(n_env)]
    index=0
    with open(args.data_txt_path,'r') as file:
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
    
    return all_samples, all_targets

def parse_data_meta_path(args):

    class_list=args.class_list.split()
    print("loading data and metadata!!!!")

    if(args.data_test_path !='' and args.data_train_path!=''):
        # input("what are we doing here?")
        _, _,train_list= list_all_files_meta(args,args.data_train_path)

        train_datasize=args.num_instance_each_class
        # args.num_instance_each_class=args.num_instance_each_class_test

        _, _,test_list= list_all_files_meta(args,args.data_test_path)
        all_list=train_list+test_list
        # args.num_instance_each_class=train_datasize
    else:
        # data_dir = args.data_folder_path
        print('parse data from {}'.format(args.data_folder_path))
        train_list, test_list,all_list= list_all_files_meta(args,args.data_folder_path)
    os.makedirs("{}/data_cache/".format(args.data_dir),exist_ok=True)
    for stage in ['train','test','all']:
        if(stage=='train'):
            image_list=train_list
        elif(stage=='test'):
            image_list=test_list
        else:
            image_list=all_list
        # folder need to be like */folder/timestamp/class/image.png
        with open('{}/data_cache/data_{}_path.txt'.format(args.data_dir,stage) , 'w') as file:
            file.write("file class_index timestamp")
            for item in image_list:
                img_path, upload_ts, taken_ts = item
                name_list=img_path.split('/')
                classes=name_list[-2]
                if classes not in class_list:
                    continue
                class_index=class_list.index(classes)
                timestamp=name_list[-3]
                file.write("\n")
                file.write(img_path+ " "+str(class_index)+" "+str(timestamp)+" "+ str(upload_ts) +" " + str(taken_ts))
        print('{} parse path finish!'.format(stage))

class Clear10Base(Dataset):
    def __init__(self, args):
        super().__init__()
        # self.data_file = f'{str(self)}.pkl'
        # preprocess(args)

        args.defrost()
        args.class_list = 'BACKGROUND baseball bus camera cosplay dress hockey laptop racing soccer sweater'
        args.data_train_path = '/projectnb/ivc-ml/amliu/clear/clear10/train/labeled_images'
        args.data_test_path = '/projectnb/ivc-ml/amliu/clear/clear10/test/labeled_images'
        args.data_folder_path = '/projectnb/ivc-ml/amliu/clear/clear10/train/labeled_images'
        args.num_instance_each_class = 300
        args.num_instance_each_class_test = 50
        args.timestamp = 10
        args.num_classes = 10
        args.ts_norm = 'minmax'
        args.root = args.data_dir
        args.data_txt_path = str(Path(args.root) / Path('data_cache/data_train_path.txt'))

        args.freeze()

        self.data_infos, self.all_targets = prepare_clear_samples(args)
        self.transform_train = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                    transforms.RandomGrayscale(p=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        # self.transform_train = transforms.Compose([
        #         transforms.Resize((224, 224)),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406],
        #                             [0.229, 0.224, 0.225])
        #     ])
        self.transform_test = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
            ])
        

        # dataset = get_dataset(dataset="fmow", root_dir=args.data_dir, download=False)
        self.datasets = {}
        random.seed(args.random_seed)  # for reproducibility
        np.random.seed(args.random_seed)
        for year in range(len(self.data_infos)):
            self.datasets[year] = {}
            self.datasets[year][0] = {}

            for year in range(len(self.data_infos)):
                # Initialize dataset structure for this year
                self.datasets[year] = {}
                self.datasets[year][0] = {'image_paths': [], 'labels': []}  # Training set
                self.datasets[year][1] = {'image_paths': [], 'labels': []}  # Validation set
                self.datasets[year][2] = {'image_paths': [], 'labels': []}  # Combined set
                
                # Get all samples for current year
                samples = self.data_infos[year]
                
                # Randomly shuffle the indices
                indices = list(range(len(samples)))
                random.shuffle(indices)
                
                # Calculate split point
                split_idx = int(len(indices) * (1 - ID_HELD_OUT))
                
                # Split into training and validation indices
                train_indices = indices[:split_idx]
                val_indices = indices[split_idx:]
                
                # Populate the datasets
                for idx in range(len(samples)):
                    img_path = samples[idx][0]
                    class_id = int(samples[idx][1]) - 1  # Convert to int and subtract 1
                    
                    # Add to combined set (mode 2)
                    self.datasets[year][2]['image_paths'].append(img_path)
                    self.datasets[year][2]['labels'].append(class_id)
                    
                    # Add to either training or validation set
                    if idx in train_indices:
                        self.datasets[year][0]['image_paths'].append(img_path)
                        self.datasets[year][0]['labels'].append(class_id)
                    else:
                        self.datasets[year][1]['image_paths'].append(img_path)
                        self.datasets[year][1]['labels'].append(class_id)
            
        self.root = args.root
        self.args = args
        self.num_classes = 10
        self.current_time = 0
        self.num_tasks = len(self.data_infos)
        self.ENV = [year for year in range(0, self.num_tasks)]
        self.resolution = 224
        self.mode = 0
        self.ssl_training = False
        self.class_id_list = {i: {} for i in range(self.num_classes)}
        self.task_idxs = {}
        start_idx = 0
        for year in self.ENV:
            end_idx = start_idx + len(self.datasets[year][self.mode]['labels'])
            self.task_idxs[year] = {}
            self.task_idxs[year] = [start_idx, end_idx]
            start_idx = end_idx
            for classid in range(self.num_classes):
                sel_idx = np.nonzero(self.datasets[year][self.mode]['labels'] == classid)[0]
                self.class_id_list[classid][year] = sel_idx
    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

    def __str__(self):
        return 'fmow'

    def update_historical(self, idx, data_del=False):
        print("domain idx", idx)
        time = self.ENV[idx]
        prev_time = self.ENV[idx - 1]
        self.datasets[time][self.mode]['image_paths'] = np.concatenate(
            (self.datasets[prev_time][self.mode]['image_paths'], self.datasets[time][self.mode]['image_paths']), axis=0)
        self.datasets[time][self.mode]['labels'] = np.concatenate(
            (self.datasets[prev_time][self.mode]['labels'], self.datasets[time][self.mode]['labels']), axis=0)
        if data_del:
            del self.datasets[prev_time]
        for classid in range(self.num_classes):
            sel_idx = np.nonzero(self.datasets[time][self.mode]['labels'] == classid)[0]
            self.class_id_list[classid][time] = sel_idx

    def update_current_timestamp(self, time):
        self.current_time = time

    def get_input(self, idx):
        """Returns x for a given idx."""
        # print(self.current_time, self.mode)
        img_path = self.datasets[self.current_time][self.mode]['image_paths'][idx]
        img = Image.open(img_path).convert('RGB')
        return img

    def get_lisa_new_sample(self, time_idx, classid, num_sample):
        idx_all = self.class_id_list[classid][time_idx]
        sel_idx = np.random.choice(idx_all, num_sample, replace=True)
        if self.mode == 0:
            image = torch.stack([self.transform_train(self.get_input(idx)) for idx in sel_idx], dim=0)
        else:
            image = torch.stack([self.transform_test(self.get_input(idx)) for idx in sel_idx], dim=0)
        # image = torch.stack([self.transform(self.get_input(idx)) for idx in sel_idx], dim=0)
        label = self.datasets[time_idx][self.mode]['labels'][sel_idx]
        return torch.FloatTensor(image).cuda(), torch.LongTensor(label).unsqueeze(1).cuda()
    
    def split_and_save_indices(self, save_path, mem_ratio = 0.1):
    
        os.makedirs(save_path, exist_ok=True)
        indices_path = os.path.join(save_path, 'mem_{}.pkl'.format(mem_ratio))
        
        if os.path.exists(indices_path):
            with open(indices_path, 'rb') as f:
                mode3_indices = pickle.load(f)
        else:
            mode3_indices = {}
            for year in sorted(self.datasets.keys()):
                total_samples = len(self.datasets[year][0]['labels'])
                num_samples = int(mem_ratio * total_samples)
                
                all_indices = np.arange(total_samples)
                np.random.shuffle(all_indices)
                mode3_indices[year] = all_indices[:num_samples]
                print(f"Year {year}: {len(mode3_indices[year])} samples selected for mode 3")
            
            with open(indices_path, 'wb') as f:
                pickle.dump(mode3_indices, f)
        
        # Update datasets with mode 3 and remove from mode 0
        for year in sorted(self.datasets.keys()):
            mode3_idx = mode3_indices[year]
            # mode0_idx = np.setdiff1d(np.arange(len(self.datasets[year][0]['labels'])), mode3_idx)
            
            # Create mode 3
            try:
                self.datasets[year][3] = {
                    'image_paths': np.array(self.datasets[year][0]['image_paths'])[mode3_idx],
                    'labels': np.array(self.datasets[year][0]['labels'])[mode3_idx]
                }
            except:
                pdb.set_trace()


class Clear10(Clear10Base):
    def __init__(self, args):
        super().__init__(args=args)
        # for mode in range(3):
        #     for idt in self.datasets.keys():
        #         print(idt, mode, len(self.datasets[idt][mode]['labels']))
        # pdb.set_trace()

    def __getitem__(self, idx):
        label_tensor = torch.LongTensor([self.datasets[self.current_time][self.mode]['labels'][idx]])

        if self.args.method in ['simclr', 'swav'] and self.ssl_training:
            image = self.get_input(idx)
            image = image.resize((self.resolution, self.resolution))
            # print(image.size)
            return image, label_tensor, ''
        else:
            if self.mode == 0:
                image_tensor = self.transform_train(self.get_input(idx))
            else:
                image_tensor = self.transform_test(self.get_input(idx))
            # image_tensor = self.transform(self.get_input(idx))
            return image_tensor, label_tensor

    def __len__(self):
        try:
            return len(self.datasets[self.current_time][self.mode]['labels'])
        except:
            pdb.set_trace()



class Clear10Group(Clear10Base):
    def __init__(self, args):
        super().__init__(args=args)
        self.group_size = args.group_size
        self.train = True
        self.num_groups = (args.split_time - args.init_timestamp + 1) - args.group_size + 1

    def __getitem__(self, idx):
        if self.mode == 0:
            np.random.seed(idx)
            # Select group ID
            idx = self.ENV.index(self.current_time)
            possible_groupids = [i for i in range(max(1, (idx + 1) - self.group_size + 1))]
            groupid = np.random.choice(possible_groupids)

            # Pick a time step in the sliding window
            window = np.arange(groupid, groupid + self.group_size)
            sel_time = self.ENV[np.random.choice(window)]
            start_idx, end_idx = self.task_idxs[sel_time]

            # Pick an example in the time step
            sel_idx = np.random.choice(np.arange(start_idx, end_idx))
            label = self.datasets[self.current_time][self.mode]['labels'][sel_idx]
            image_tensor = self.transform_train(self.get_input(sel_idx))
            label_tensor = torch.LongTensor([label])
            group_tensor = torch.LongTensor([groupid])

            del groupid
            del window
            del sel_time
            del start_idx
            del end_idx
            del sel_idx
            return image_tensor, label_tensor, group_tensor
        else:
            image_tensor = self.transform_test(self.get_input(idx))
            label = self.datasets[self.current_time][self.mode]['labels'][idx]
            label_tensor = torch.LongTensor([label])
            return image_tensor, label_tensor

    def group_counts(self):
        idx = self.ENV.index(self.current_time)
        return torch.LongTensor([1 for _ in range(min(self.num_groups, idx + 1))])

    def __len__(self):
        return len(self.datasets[self.current_time][self.mode]['labels'])




def preprocess(args):
    if not os.path.isfile(os.path.join(args.data_dir, 'fmow.pkl')):
        raise RuntimeError(args.data_dir, "dataset fmow.pkl is not yet ready! Please download from   https://drive.google.com/u/0/uc?id=1s_xtf2M5EC7vIFhNv_OulxZkNvrVwIm3&export=download   and save it as fmow.pkl")
