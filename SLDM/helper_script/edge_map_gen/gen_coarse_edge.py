from utils.cityscape_ds_alpha import load_data
import os
import torch
import cv2

def preprocess_edge(inst_map, filename, subset):
    # utils to get the edge of image
    def get_edges(t):
        # zero tensor, prepare to fill with edge (1) and bg (0)
        edge = torch.ByteTensor(t.size()).zero_().to(t.device)
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1]).byte()
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1]).byte()
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :]).byte()
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :]).byte()
        return edge.float()
    
    instance_edge_map = get_edges(inst_map)
    # saving coarse-edge-image
    for inst_msk, fn_w_ext in zip(instance_edge_map, filename):
        fn_lst = fn_w_ext.split('_')
        # DA_Data
        if len(fn_lst) == 1:
            fn_name = fn_lst[0].split('.')[0]
            prefix = f"/data1/dataset/Cityscapes/gtFine/{subset}/DA_Data/"
            path = prefix + f"{fn_name}_crEdgeMaps.png"
        # general subset
        elif len(fn_lst) == 4:
            #city, *id_info = fn_lst
            prefix = f"/data1/dataset/Cityscapes/gtFine/{subset}/{fn_lst[0]}/"
            fn_lst[-1] = 'gtFine_crEdgeMaps.png'
            tmp_path = "_".join(fn_lst)
            path = prefix + tmp_path
        else:
            raise ValueError

        #breakpoint()
        #               ( 1 x H x W -> H x W ) 
        cv2.imwrite(path, inst_msk.numpy()[0]*255.)
        


def get_cfg_str():
    return '''
    gen_set = 'train'
    [dataset]
        data_dir = '/data1/dataset/Cityscapes'
        resize_size = 540
        subset_type = 'all' 
        fn_qry = '*/*.png'
        random_flip = False@bool
        ret_dataset = False@bool

        [dataset.data_ld_kwargs]
            shuffle = False@bool
            batch_size = 8
            num_workers = 6
    '''


if __name__ == "__main__":
    from easy_configer.Configer import Configer
    cfger = Configer()
    cfger.cfg_from_str( get_cfg_str() )

    data_ld = load_data(**cfger.dataset)
    
    # train subset 
    for idx, batch in enumerate(data_ld[cfger.gen_set], 0):
        preprocess_edge(batch["segmap"]['instance'], batch['filename'], cfger.gen_set)

    print('done!')
    # should have 3975..
    #print(f"total mask num : {len(all_lst)}\n")
