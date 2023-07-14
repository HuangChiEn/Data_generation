#export PL_RECONCILE_PROCESS=1
python  train_compress_model.py --base compression_module/configs/cityscape_vqgan.yaml -t True --gpus 1