import pickle
import cv2
import os
import gzip
import time
from remote_plot import plt

with open('./cityscapes-to-coco-conversion/un_gtFine_train_bbox.pkl', 'rb') as f_ptr:
    tra_dict = pickle.load(f_ptr)

for idx, im_inst in enumerate(tra_dict.values()):
    im_path = os.path.join("/data1/dataset/Cityscapes", im_inst['fn'])
    img = cv2.imread(im_path)
    if idx < 46:
        continue
    #print(idx)
    #plt.imshow(img)
    breakpoint()
    #continue

    for inst in im_inst['info']:
        x, y, w, h = inst['bbox']
        xy_min = (x, y)
        xy_max = (x+w, y+h)
        if not inst['iscrowd']:
            cv2.rectangle(img, xy_min, xy_max, (0, 255, 0), 8)
        
        plt.imshow(img)
        breakpoint()
    
    # subplot(r,c) provide the no. of rows and columns
    #breakpoint()
    #plt.subplot(121)
    #plt.imshow(img)
    #plt.subplot(122)
    #plt.imshow(im_inst['msk'])
    #time.sleep(12)
