import json
import pickle
import cv2
import numpy as np
import torch
import gzip

# TODO : estimate yaw angle
##TODO : plot 3D cityscape bbox center to the 2D image, then putText to locate the instanceID and scurrnt json file
#from math import atan2, asin
#q0, q1, q2, q3 = inst_ann['rotation']
#roll  = atan2(2.0 * (q.q3 * q.q2 + q.q0 * q.q1) , 1.0 - 2.0 * (q.q1 * q.q1 + q.q2 * q.q2));
#pitch = asin(2.0 * (q.q2 * q.q0 - q.q3 * q.q1));
#yaw   = atan2(2.0 * (q.q3 * q.q0 + q.q1 * q.q2) , - 1.0 + 2.0 * (q.q0 * q.q0 + q.q1 * q.q1));


with open('./instancesonly_filtered_gtFine_train.json') as inst_ptr:
    inst_js = json.load(inst_ptr)

id_lst = [ im_inst['id'] for im_inst in inst_js['images'] ]
tra_dict = {}.fromkeys(id_lst)

total_len = len(inst_js['annotations'])-1
for idx, inst_ann in enumerate(inst_js['annotations']):
    im_id = inst_ann['image_id']
    im_ann = inst_js['images'][im_id]

    # multiple objects in the same image share the same meta-info
    if tra_dict[im_id] == None:
        # make binary mask for car classes : 
        tmp = im_ann['file_name'].replace('leftImg8bit', 'gtFine')
        label_path = tmp.replace('_gtFine', '_gtFine_labelIds')
        label_map = cv2.imread('/data1/dataset/Cityscapes/'+label_path, cv2.IMREAD_GRAYSCALE)
        
        h, w = label_map.shape
        label_map = torch.from_numpy(label_map[None, None, ...]).type(torch.int64)
        input_label = torch.FloatTensor(1, 34, h, w).zero_().to(label_map.device)
        input_semantics = input_label.scatter_(1, label_map, 1.0).numpy()

        tra_dict[im_id] = {
            'msk' : input_semantics[0, 26, ...], 
            'fn' : im_ann['file_name'], 
            'info' : []
        }

    info_dict = {}
    info_dict['bbox'] = x, y, w, h = inst_ann['bbox']
    info_dict['ratio'] = w / h
    info_dict['iscrowd'] = inst_ann['iscrowd']
    tra_dict[im_id]['info'].append(info_dict)
    
    print(f"{idx} / {total_len}")

train_dict = {}
for k, v in tra_dict.items():
    if v != None:
        train_dict[k] = v

with open('./un_gtFine_train_bbox.pkl', 'wb+') as sav_ptr:
    pickle.dump(train_dict, sav_ptr)
