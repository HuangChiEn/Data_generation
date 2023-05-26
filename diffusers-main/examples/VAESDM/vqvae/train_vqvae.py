


def main(cfger):
    ...



## TODO: cp & past cfg from https://raw.githubusercontent.com/CompVis/taming-transformers/master/configs/imagenet_vqgan.yaml
def get_cfg_str():
    return '''
    seed = 42@int
    ...
    
    [dataloader]
        data_dir = /data1/dataset/Cityscapes@str
        image_size = 270@int
        batch_size = 8@int
        num_workers = 1@int

    '''


if __name__ == "__main__":
    from easy_configer.Configer import Configer
    cfger = Configer()
    cfger.cfg_from_str( get_cfg_str() )

    main(cfger)
