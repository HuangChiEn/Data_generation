import cv2
import numpy as np
import random
import os
import pickle

class Data_augmentation():
    def __init__(self, car_dir = None, inst_pkl_path = None):
        # get all car image path
        self.world = os.listdir(car_dir)
        self.car_dir = car_dir
        # with open(inst_pkl_path, 'rb+') as f_ptr:
        #     self.inst_dict = pickle.load(f_ptr)

    # Note : inplace operation
    def remove_overlap_bbox(self, inst_lst):
        # https://stackoverflow.com/questions/20925818/algorithm-to-check-if-two-boxes-overlap
        def isOverlapping2D(box_xy1, box_xy2):
            (xmin1, xmax1, ymin1, ymax1) = box_xy1
            (xmin2, xmax2, ymin2, ymax2) = box_xy2

            # chk x-axis proj overlapped & chk y-axis proj overlapped
            if (xmax1 >= xmin2 and xmax2 >= xmin1) and (ymax1 >= ymin2 and ymax2 >= ymin1):
                return True
            # or it will be non-overlap
            return False

        # make bin for counting num of bbox overlapped with the others
        n_bbox = len(inst_lst['bbox'])
        bbox_bin = [0 for _ in n_bbox]

        # prepare traversal index pairs (flatten the nested for-loop)
        pair_idx = [(i, j, 0) for j in range(n_bbox) for i in range(j)]

        for z, (idx, jdx, _) in enumerate(pair_idx):
            i_bbox, j_bbox = inst_lst[idx]['bbox'], inst_lst[jdx]['bbox']
            # unroll idx bbox cords
            i_xmin, i_ymin, w, h = i_bbox
            i_xmax, i_ymax = i_xmin + w, i_ymin + h
            # unroll jdx bbox cords
            j_xmin, j_ymin, w, h = j_bbox
            j_xmax, j_ymax = j_xmin + w, j_ymin + h

            if isOverlapping2D((i_xmin, i_xmax, i_ymin, i_ymax),
                               (j_xmin, j_xmax, j_ymin, j_ymax)):
                bbox_bin[idx] += 1
                bbox_bin[jdx] += 1
                # mark idx, jdx bbox is overlapped
                pair_idx[z][2] += 1

        # duplicate counting and remove bbox procedure..
        sort_idx = [i[0] for i in sorted(enumerate(bbox_bin), key=lambda x: x[1])]

        # sort_idx[-1] is max_idx, we discard the car bbox \
        #   which overlap with the other bboxs with most times
        for max_idx in sort_idx.reverse():
            # discard the car bbox
            bbox_bin[max_idx] = 0 ; inst_lst.pop(max_idx)
            # boardcast "the max_idx-th car bbox is discarded" to the other car's bbox
            for z, (idx, jdx, cnt) in enumerate(pair_idx):
                # if those 2 box is not overlapped ~
                if (cnt == 0) and (max_idx != idx) and (max_idx != jdx):
                    continue

                rm_idx = jdx if max_idx == idx else idx
                bbox_bin[rm_idx] -= 1
                pair_idx[z][2] -= 1

            # now, how about the other car bbox, does those bbox still overlap with each other ?
            if sum(bbox_bin) == 0:
                break  # if so, mission complete ~

    def augment_sample(self):
        for im_inst in self.inst_dict.values():
            # read img, binary msk
            im_path = os.path.join("/data1/dataset/Cityscapes", im_inst['fn'])
            img, msk = cv2.imread(im_path), im_inst['msk']

            # eliminate mutual covered bbox..
            self.remove_overlap_bbox(im_inst['info'])

            # for each img traversal each objects
            for inst in im_inst['info']:
                x, y, w, h = inst['bbox']
                if not inst['iscrowd']:
                    crop_img, crop_msk = img[y:y + h, x:x + w], msk[y:y + h, x:x + w]
                    new_im, _ = self.find_best_object(crop_img, crop_msk)
                    img[y:y + h, x:x + w] = new_im if new_im else crop_img

            cv2.imwrite('test.png', img)
            # don't forgot to take a break ~
            break

    def IOU(self, mask1, mask2):
        # caculate the IOU
        h = int((mask1.shape[0] + mask2.shape[0]) / 2)
        w = int((mask1.shape[1] + mask2.shape[1]) / 2)
        mask1 = cv2.resize(mask1, (h, w))
        mask2 = cv2.resize(mask2, (h, w))
        area1 = mask1.sum()
        area2 = mask2.sum()
        inter = ((mask1 + mask2) == 2).sum()
        mask_iou = inter / (area1 + area2 - inter)
        return mask_iou

    def H_W_ratio(self, mask1, mask2):
        # caculate the shape ratio match rate
        r1 = mask1.shape[0] / mask1.shape[1]
        r2 = mask2.shape[0] / mask2.shape[1]
        if r1 > r2:
            return r2 / r1
        return r1 / r2

    def get_group(self, group_size=100):
        # get the subset from world
        if group_size > len(self.world):
            group = list(enumerate(self.world))
        else:
            group = random.sample(list(enumerate(self.world)), group_size)
        index = []
        path = []
        for g in group:
            index.append(g[0])
            path.append(g[1])
        return index, path

    def find_best_object(self, mask, iou_th=0.9, ratio_th=0.9):
        max_image, max_mask, max_index, max_IOU = None, None, None, 0
        Id, path = self.get_group()
        for index, target_image_path in zip(Id, path):
            p = self.car_dir[:-5] + "mask\\" + target_image_path
            target_mask = cv2.imread(os.path.join(p), 0)
            if self.H_W_ratio(mask, target_mask) >= ratio_th:
                iou = self.IOU(mask, target_mask)
                if iou >= iou_th and iou > max_IOU:
                    max_IOU = iou
                    max_mask = target_mask
                    max_index = index
        if max_mask is not None:
            max_image = cv2.imread(self.car_dir + "\\" + self.world[max_index])
            del self.world[max_index]
            return max_image, max_mask
        else:
            return None, None

    def filling_car(self, image1, image2, mask1, mask2):
        image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))
        mask1 = cv2.resize(mask1, (image2.shape[1], image2.shape[0]))
        mask1 = np.expand_dims(mask1, axis=-1)
        mask2 = np.expand_dims(mask2, axis=-1)
        result_image = image1 * mask2 + image2 * (1 - mask1)
        #
        # cv2.imshow("result_image",result_image)
        # cv2.imshow("image1", image1)
        # cv2.imshow("image2", image2)
        #
        # cv2.imshow("mask1", mask1*255)
        # cv2.imshow("mask2", mask2*255)
        return result_image

def main():
    #mask1 = cv2.imread(r"C:\Users\ASUS\Desktop\mask\Mitsubishi$$Shogun Sport$$2004$$Silver$$62_20$$18$$image_4.png", 0)
    mask2 = cv2.imread(r"C:\Users\ASUS\Desktop\Renault$$Koleos$$2017$$Blue$$75_10$$44$$image_3mask.png", 0)
    #image1 = cv2.imread(r"C:\Users\ASUS\Desktop\image\Mitsubishi$$Shogun Sport$$2004$$Silver$$62_20$$18$$image_4.png")
    image2 = cv2.imread(r"C:\Users\ASUS\Desktop\Renault$$Koleos$$2017$$Blue$$75_10$$44$$image_3.png")
    DA = Data_augmentation(car_dir = r"C:\Users\ASUS\Desktop\image")
    print(len(DA.world))
    max_image, max_mask = DA.find_best_object(mask2)
    print(max_image.shape)
    print(len(DA.world))
    cv2.imshow("result_image",max_image)
    # cv2.imshow("image1", image1)
    # cv2.imshow("image2", image2)
    #
    # cv2.imshow("mask1", mask1*255)
    # cv2.imshow("mask2", mask2*255)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()