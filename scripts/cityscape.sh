#NCU cityscape : /media/harry/f7a9be3d-25cd-45d6-b503-7cb8bd32dbd5/cityscape_synthetic
MPI_HOME=/home/harry/openmpi
export PATH=${MPI_HOME}/bin:$PATH
export LD_LIBRARY_PATH=${MPI_HOME}/lib:$LD_LIBRARY_PATH
export MANPATH=${MPI_HOME}/share/man:$MANPATH

#Training
export OPENAI_LOGDIR='../OUTPUT/Cityscapes540-VAESDM-LDM84ch-256CH-500ksteptest' # --attention_resolutions 32,16,8 --lr_warmup_steps 10000
mpiexec -np 1 --allow-run-as-root python ../image_train.py --data_dir /data1/dataset/Cityscapes --dataset_mode cityscapes --lr 1e-4 --batch_size 48 --attention_resolutions 32,16,8 --diffusion_steps 1000 \
                                   --image_size 540 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 8 --num_res_blocks 2  \
                                   --resblock_updown True --use_fp16 True --use_scale_shift_norm True --use_checkpoint True --num_classes 34 \
	                                 --class_cond True --no_instance False --training_step 500000 --use_vae True --mask_emb 'resize'  \
	                                 --catch_path /data/harry/Cityscape_catch/VAE_540_label_resize #--resume_checkpoint ../OUTPUT/Cityscapes540-VAESDM-maskk_latent-256CH-500kstep/model050000.pt \

# Classifi er-free Finetune
#export OPENAI_LOGDIR='../OUTPUT/Cityscapes-SDM-256CH-10epoch-FINETUNE'
#mpiexec --allow-run-as-root -np 4 python ../image_train.py --data_dir /data1/dataset/Cityscapes --dataset_mode cityscapes --lr 2e-5 --batch_size 4 --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 256 --learn_sigma True \
#	     --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --use_checkpoint True --num_classes 34 \
#	     --class_cond True --no_instance False --drop_rate 0.2 --resume_checkpoint ../OUTPUT/Cityscapes-SDM-256CH-10epoch/model002000.pt --training_step 2000


# Testing
#export OPENAI_LOGDIR='../OUTPUT/Cityscapes540-SDM-256CH-160kstep-Test'
#mpiexec -np 1 --allow-run-as-root python ../image_sample.py --data_dir /data1/dataset/Cityscapes --dataset_mode cityscapes --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 270 --learn_sigma True \
#       --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --num_classes 34 \
#       --class_cond True --no_instance False --batch_size 8 --num_samples 500 --model_path ../OUTPUT/Cityscapes270-SDM-256CH-500epoch/model160000.pt --results_path ../RESULTS/Cityscapes540-SDM-256CH-160kstep --s 1 --use_vae False \
#       --mask_emb 'resize'