# VAE Semantic Image Synthesis via Diffusion Models (VAE-SDM)

### Code base movement
#### TKS to harry, the SDM-LDM and their baseline (SDM) have been integrated into the hugging face example, where located in `./diffusers-main/examples/VAESDM` . 

### Todo

- [x] Train the SDE with the CityScape Dataset
  - [x] Implementation of the Cityscape results from the SDM paper
  - [x] Training SDM with the 270X360 @harry
- [x] Adding the VAE(VQVAE)   
  - [x] Codding in training @harry
  - [x] Codding in inference @harry
- [ ] VQ-VAE and VAE training code
  - [ ] VAE training code
  - [X] VQ-VAE training code
  - [ ] fix multi GPU training
- [x] Optimize code
  - [x] Refactor the official code (Change to the Hugging Face)@harry @HuangChiEn



[//]: # (&nbsp;)

[//]: # ()
[//]: # (<img src='assets\results.png' align="left">  )

[//]: # ()
[//]: # (&nbsp;)

[//]: # ()
[//]: # (<img src='assets/diffusion.png' align="left">)

[//]: # ()
[//]: # (&nbsp;)

[//]: # ()

### [Reference Paper](https://arxiv.org/abs/2207.00050) : Semantic Image Synthesis via Diffusion Models (SDM)

### [Reference Code](https://github.com/WeilunWang/semantic-diffusion-model) : https://github.com/WeilunWang/semantic-diffusion-model

## SDM Example Results
* **Cityscapes:**

![image](example/cityspace.png)

## Prerequisites
- Linux
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN
- MPI([openmpi](https://www.open-mpi.org/) Link:[how install](https://blog.csdn.net/liu_feng_zi_/article/details/107429347) better than [mpich](https://www.mpich.org/downloads/) Link:[how install](https://cloud.tencent.com/developer/article/2111003))

## Dataset Preparation
TKS to HuangChiEn, train diffusion model please use the  [cityscape_ds_alpha.py](diffusers-main/VAESDM/cityscape_ds_alpha.py) dataset, and train the VQVAE please use the [cityscape_ds.py](diffusers-main/VQVAE/taming/data/cityscape_ds.py).

### NEGCUT Training and Test

- Download the dataset.

- Train the VQ-SDM model on our code(Hugging Face):
```bash
cd diffusers-main/VAWSDM
accelerate launch --mixed_precision="fp16" SDM_LDM.py --gradient_checkpointing --use_ema --output_dir ./SDM_LDM
```

- Train the SDM model on our code(Hugging Face):
```bash
cd diffusers-main/VAWSDM
accelerate launch --mixed_precision="fp16" SDM.py --gradient_checkpointing --use_ema --output_dir ./SDM
```

- Test the SDM or VQ-SDM model on our code(Hugging Face):
```bash
cd diffusers-main/VAWSDM
python image_generation.py
```


- Train the SDM model on official code:
```bash
export OPENAI_LOGDIR='./OUTPUT/Cityscapes-SDM-256CH'
mpiexec -n 4 python image_train.py --data_dir /data1/dataset/Cityscapes --dataset_mode cityscapes --lr 1e-4 --batch_size 4 --attention_resolutions 32,16,8 --diffusion_steps 1000 \
                                   --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2  \
                                   --resblock_updown True --use_fp16 True --use_scale_shift_norm True --use_checkpoint True --num_classes 35 \
	                           --class_cond True --no_instance False
```

- Test the SDM model on official code:
```bash
export OPENAI_LOGDIR='../OUTPUT/Cityscapes360-SDM-256CH-12kstep-TEST'
mpiexec --allow-run-as-root -np 4 python ../image_sample.py --data_dir /data1/dataset/Cityscapes --dataset_mode cityscapes --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 360 --learn_sigma True \
       --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --num_classes 34 \
       --class_cond True --no_instance False --batch_size 4 --num_samples 8 --model_path ../OUTPUT/Cityscapes360-SDM-256CH-500epoch/model012000.pt --results_path ../RESULTS/Cityscapes360-SDM-256CH-12kstep --s 1.5
```

Please refer to the 'scripts/cityscap.sh' for more details.

### Apply a pre-trained NEGCUT model and evaluate

#### Official pretrained Models (to be updated)
|Dataset       | Download link                                                                                                                                                                                            |
|:-------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|Cityscapes| [Checkpoint(our)](http://140.115.53.100:1118/sharing/tyD0tSSyT) \| [Visual results](https://drive.google.com/file/d/1TbLGCFJqRI4E8pFZJoHmj8MgDbwtjzhP/view?usp=sharing)                                  |
|ADE20K| [Checkpoint](https://drive.google.com/file/d/1O8Avsvfc8rP9LIt5tkJxowMTpi1nYiik/view?usp=sharing) \| [Visual results](https://drive.google.com/file/d/1NIXmrlBHqgyMHAoLBlmU8YELmL8Ij4kV/view?usp=sharing) |
|CelebAMask-HQ | [Checkpoint](https://drive.google.com/file/d/1iwpruJ5HMHdAA1tuNR8dHkcjGtxzSFV_/view?usp=sharing) \| [Visual results](https://drive.google.com/file/d/1NDfU905iJINu4raoj4JdMOiHP8rTXr_M/view?usp=sharing) |
|COCO-Stuff | [Checkpoint](https://drive.google.com/file/d/17XhegAk8V5W3YiFpHMBUn0LED-n7B44Y/view?usp=sharing) \| [Visual results](https://drive.google.com/file/d/1ZluvN9spJF8jlXlSQ98ekWTmHrzwYCqo/view?usp=sharing) |

#### Our pretrained diffusion Models (to be updated)
| Method(v0.1 is mean the offical code)   | Compress model  | Dataset    | Download link                                              |
|:----------------------------------------|:----------------|:-----------|:-----------------------------------------------------------|
| SDM 270*360 v0.1                        | None            | Cityscapes | [Checkpoint](http://140.115.53.100:1118/sharing/Gga81n9Hy) |
| VAE-SDM 540*720 v0.1                    | Pretrain VAE    | Cityscapes | [Checkpoint](http://140.115.53.100:1118/sharing/H6Ko18k7k) |
| VQ-SDM 540*720 v0.9 (VQ 46epoch)        | SPAD VQ V1(our) | Cityscapes | [Checkpoint](http://140.115.53.100:1118/sharing/PgXBZ4zmR) |
| VQ-SDM 540*720 v1.0 (VQ 70epoch)        | SPAD VQ V1(our) | Cityscapes | [Checkpoint](http://140.115.53.100:1118/sharing/pLaA1eTxo) |
| VQ-SDM-RESAIL 540*720 v1.0 (VQ 70epoch) | SPAD VQ V1(our) | Cityscapes | [Checkpoint](http://140.115.53.100:1118/sharing/PS47oQsjK) |
| VQ-SDM-SPM 540*720 v1.0 (VQ 70epoch)    | SPAD VQ V1(our) | Cityscapes | [Checkpoint](http://140.115.53.100:1118/sharing/RFj15T5N4) |
 
#### Our pretrained VQ Models (to be updated)
| Model        | Dataset     | Download link  |
|:-------------|:------------|:---------------|
| VQ_VAE       | Cityscapes  | [Checkpoint](http://140.115.53.100:1118/sharing/okNuC3ju5) |
| SPADE-VQ_VAE | Cityscapes  | [Checkpoint](http://140.115.53.100:1118/sharing/Xxe2UccPD) |


- To evaluate the model (e.g., ADE20K), first generate the test results:
- To calucate FID metric, you should update "path1" and "path2" in "evaluations/test_with_FID.py" and run:
```bash
python evaluations/test_with_FID.py
```

- To calcuate LPIPS, you should evaluate the model for 10 times and run:
```bash
python evaluations/lpips.py GENERATED_IMAGES_DIR
```

### Acknowledge
Our code is developed based on
[SDM](https://github.com/WeilunWang/semantic-diffusion-model)
-  We also thank [guided-diffusion](https://github.com/openai/guided-diffusion). "test_with_FID.py" in [OASIS](https://github.com/boschresearch/OASIS) for FID computation, "lpips.py" in [stargan-v2](https://github.com/clovaai/stargan-v2) for LPIPS computation.

