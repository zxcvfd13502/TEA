
import os
import os.path as osp
import json
from sklearn.model_selection import train_test_split
from dateutil import parser
from datetime import datetime
import time
# data_dir ='/scratch/zhiqiu/yfcc_dynamic_10/dynamic_300/images'


# def list_all_files(rootdir):
#     _files = []
#     list_file = os.listdir(rootdir)
#     for i in range(0,len(list_file)):
#         path = os.path.join(rootdir,list_file[i])

#         if os.path.isdir(path):
#             _files.extend(list_all_files(path))
#         if os.path.isfile(path):
#              _files.append(path)
#     return _files

def list_all_files(args,rootdir):
    print("listing all files in rootdir:", rootdir)
    # input()
    train_list,test_list,all_list = [],[],[]
    bucket_list = os.listdir(rootdir)
    # bucket_list=list(filter(lambda a: 'bucket_' in a,bucket_list))
    if('0' in bucket_list):
        bucket_list.remove('0') # skip bucket 0, since it's for pretrain feature
    classes_list=  os.listdir(osp.join(rootdir,bucket_list[0]))
    if('clear25d' in args.split and 'BACKGROUND' in classes_list):
        classes_list.remove('BACKGROUND') # skip bucket 0, since it's for pretrain feature
    for bucket in bucket_list:
        for classes in classes_list:
            image_list=os.listdir(osp.join(rootdir,bucket,classes))
            image_list=list(map(lambda a: osp.join(osp.join(rootdir,bucket,classes,a)), image_list))
            image_list=image_list[:args.num_instance_each_class] # if background class have more image, we use only part of it
            # print("image_list", image_list)
            try:
                assert len(image_list)==args.num_instance_each_class
            except:
                # import pdb;pdb.set_trace()
                print("the bucket", str(bucket), 'class', str(classes), 'has', len(image_list), 'not', args.num_instance_each_class)
                # exit()
            train_subset,test_subset=train_test_split(image_list,test_size=args.test_split, random_state=args.seed)
            train_list.extend(train_subset)
            test_list.extend(test_subset)
            all_list.extend(image_list)
    import random
    random.seed(args.seed)
    random.shuffle(train_list)
    random.shuffle(test_list)
    random.shuffle(all_list)
    return train_list,test_list,all_list

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
            train_subset,test_subset=train_test_split(image_list,test_size=args.test_split, random_state=args.seed)
            train_list.extend(get_ts(train_subset, meta_data))
            test_list.extend(get_ts(test_subset, meta_data))
            all_list.extend(get_ts(image_list, meta_data))

    import random
    random.seed(args.seed)
    random.shuffle(train_list)
    random.shuffle(test_list)
    random.shuffle(all_list)
    return train_list,test_list,all_list

def parse_data_path(args):

    class_list=args.class_list.split()
    print(class_list)
    print(args.data_test_path, args.data_train_path)
    input()
    # if available, use pre-split train/test, else, auto split the data_folder_path
    if(args.data_test_path !='' and args.data_train_path!=''):
        input("what are we doing here?")
        _, _,train_list= list_all_files(args,args.data_train_path)

        train_datasize=args.num_instance_each_class
        args.num_instance_each_class=args.num_instance_each_class_test

        _, _,test_list= list_all_files(args,args.data_test_path)
        all_list=train_list+test_list
        args.num_instance_each_class=train_datasize
    else:
        data_dir = args.data_folder_path
        print('parse data from {}'.format(data_dir))
        train_list, test_list,all_list= list_all_files(args,data_dir)
    os.makedirs("./{}/data_cache/".format(args.split),exist_ok=True)
    for stage in ['train','test','all']:
        if(stage=='train'):
            image_list=train_list
        elif(stage=='test'):
            image_list=test_list
        else:
            image_list=all_list
        # folder need to be like */folder/timestamp/class/image.png
        with open('./{}/data_cache/data_{}_path.txt'.format(args.split,stage) , 'w') as file:
            file.write("file class_index timestamp")
            for item in image_list:
                name_list=item.split('/')
                classes=name_list[-2]
                if classes not in class_list:
                    continue
                class_index=class_list.index(classes)
                timestamp=name_list[-3]
                # timestamp=name_list[-3].split('_')[-1] # since name is bucket_x
                file.write("\n")
                file.write(item+ " "+str(class_index)+" "+str(timestamp))
        print('{} parse path finish!'.format(stage))

def parse_data_meta_path(args):

    class_list=args.class_list.split()
    print("loading data and metadata!!!!")
    # print(class_list)
    # print(args.data_test_path, args.data_train_path)
    # meta_data_path = args.meta_data_path
    # meta_data = json.load(open(meta_data_path))

    # input()
    # if available, use pre-split train/test, else, auto split the data_folder_path
    if(args.data_test_path !='' and args.data_train_path!=''):
        # input("what are we doing here?")
        _, _,train_list= list_all_files_meta(args,args.data_train_path)

        train_datasize=args.num_instance_each_class
        args.num_instance_each_class=args.num_instance_each_class_test

        _, _,test_list= list_all_files_meta(args,args.data_test_path)
        all_list=train_list+test_list
        args.num_instance_each_class=train_datasize
    else:
        data_dir = args.data_folder_path
        print('parse data from {}'.format(data_dir))
        train_list, test_list,all_list= list_all_files_meta(args,data_dir)
    os.makedirs("{}/data_cache/".format(args.out_root),exist_ok=True)
    for stage in ['train','test','all']:
        if(stage=='train'):
            image_list=train_list
        elif(stage=='test'):
            image_list=test_list
        else:
            image_list=all_list
        # folder need to be like */folder/timestamp/class/image.png
        with open('{}/data_cache/data_{}_path.txt'.format(args.out_root,stage) , 'w') as file:
            file.write("file class_index timestamp")
            for item in image_list:
                img_path, upload_ts, taken_ts = item
                name_list=img_path.split('/')
                classes=name_list[-2]
                if classes not in class_list:
                    continue
                class_index=class_list.index(classes)
                timestamp=name_list[-3]
                # timestamp=name_list[-3].split('_')[-1] # since name is bucket_x
                # print(upload_ts)
                # print(taken_ts)
                # print(timestamp, datetime.utcfromtimestamp(int(upload_ts)), datetime.utcfromtimestamp(int(taken_ts)))
                file.write("\n")
                file.write(img_path+ " "+str(class_index)+" "+str(timestamp)+" "+ str(upload_ts) +" " + str(taken_ts))
        print('{} parse path finish!'.format(stage))
