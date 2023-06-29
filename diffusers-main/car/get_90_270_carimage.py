# import pandas as pd
import os
import cv2
import numpy as np
from tqdm import tqdm
from threading import Thread

# import blobfile as bf
import random

#car_ = pd.read_csv('Image_table.csv')
def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)

def get_ret_image(img_le):
    rect_le = (10, 10, img_le.shape[1], img_le.shape[0])
    mask_le = np.zeros(img_le.shape[:2], np.uint8)
    bgd_Model = np.zeros((1, 65), np.float64)
    fgd_Model = np.zeros((1, 65), np.float64)
    cv2.grabCut(img_le, mask_le, rect_le, bgd_Model, fgd_Model, 5, cv2.GC_INIT_WITH_RECT)

    #mask_le2 = get_mask_threshold(img_le)
    mask_le2 = np.where((mask_le == 0) | (mask_le == 2), 0, 1).astype('uint8')  # 做一個只有0 & 1新的mask


    #print(np.unique(mask_le2))
    img_copy_le = img_le.copy()
    img_grabcut_le = img_copy_le * mask_le2[:, :, np.newaxis]
    # cv2.imshow(f"Image", img_grabcut_le)
    # cv2.imshow(f"Mask", mask_le2 * 255)

    contours, hierarchy = cv2.findContours(mask_le2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        contours, _ = sort_contours(contours)
        #print(contours)
        # color = ['red', 'green']
        # for i in range(0, len(contours)):
        # cv2.boundingRect: 透過輪廓找到外接矩形
        # 輸出：(x, y)矩形左上角座標、w 矩形寬(x軸方向)、h 矩形高(y軸方向)
        x, y, w, h = cv2.boundingRect(contours[0])
        # for c in contours:
        #     x, y, w, h = cv2.boundingRect(c)
        #     if w > 20 or h > 20:
        #         break
        # cv2.imshow(f"Cut", img_grabcut_le[y:y + h, x:x + w])
        #
        # cv2.imshow("Original", img_le)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return img_grabcut_le, img_grabcut_le[y:y + h, x:x + w], mask_le2, mask_le2[y:y + h, x:x + w]
    else:
        return None, None, None, None




# path_list = []
# file = open("filter_file_name.txt", "w")
# for im in car_.iloc:
#     if im[3] == 90 or im[3] == 270:
#         p = [t for t in im[2].split("$$")][:-3]
#         p.append(im[2])
#         p = os.path.join(*p)
#         path_list.append(p)
#         file.write(p+"\n")
#         p = os.path.join("resized_DVM", p)
#         print(p)
# file.close()

# get file name
result_path = "/data/harry/Data_generation/diffusers-main/car/preproc_car"
file = open("filter_file_name.txt", "r")
path_list = file.readlines()
file.close()
path_list = random.sample(path_list, 120016)

def job(paths):
    for p in tqdm(paths):
        p = os.path.join(r"resized_DVM", p[:-1])
        image = cv2.imread(p)
        im1, im2, msk1, msk2 = get_ret_image(image)
        if im1 is not None:
            if im2.shape[0] > 20 or im2.shape[1] > 20:
                cv2.imwrite(os.path.join(result_path, "image", p.split("/")[-1][:-3] + "png"), im2)
                cv2.imwrite(os.path.join(result_path, "mask", p.split("/")[-1][:-3] + "png"), msk2)

num_thread = 16
thread_list = []
job_iter = int(len(path_list)/num_thread)
for i in range(num_thread-1):
    t = Thread(target=job,args=([path_list[job_iter*i:job_iter*i+job_iter]]))
    thread_list.append(t)
thread_list.append(Thread(target=job,args=([path_list[job_iter*(num_thread-1):]])))

for j in thread_list:
    j.start()

for j in thread_list:
    j.join()

print("Done")

# image = cv2.imread(r"C:\Users\ASUS\Desktop\Abarth$$595C$$2016$$White$$2_5$$13$$image_3.jpg")
# im1, im2, msk1, msk2 = get_ret_image(image)
# cv2.imshow("im2",im2)
# cv2.imshow("msk2",msk1*255)
# cv2.waitKey(0)
# cv2.destroyAllWindows()