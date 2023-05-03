#export MPIPATH=/home/harry/mpi
#export MPIPATHBIN=$MPIPATH/bin
#export MPIPATHINCLUDE=$MPIPATH/include
#export MPIPATHLIB=$MPIPATH/lib
#export MPIPATHSHARE=$MPIPATH/share
#export PATH=$PATH:$MPIPATHBIN:$MPIPATHINCLUDE:$MPIPATHLIB:$MPIPATHSHARE

#NCU cityscape : /media/harry/f7a9be3d-25cd-45d6-b503-7cb8bd32dbd5/cityscape_synthetic
#MPI_HOME=/home/harry/openmpi
#
#export PATH=${MPI_HOME}/bin:$PATH
#export LD_LIBRARY_PATH=${MPI_HOME}/lib:$LD_LIBRARY_PATH
#export MANPATH=${MPI_HOME}/share/man:$MANPATH

#Training
#export OPENAI_LOGDIR='../OUTPUT/Cityscapes270-VAESDM-128CH-500epoch'
#mpiexec -np 2 python ../image_train.py --data_dir /media/harry/f7a9be3d-25cd-45d6-b503-7cb8bd32dbd5/cityscape_synthetic --dataset_mode cityscapes --lr 1e-4 --batch_size 24 --attention_resolutions 32,16,8 --diffusion_steps 1000 \
#                                   --image_size 270 --learn_sigma True --noise_schedule linear --num_channels 128 --num_head_channels 64 --num_res_blocks 2  \
#                                   --resblock_updown True --use_fp16 True --use_scale_shift_norm True --use_checkpoint True --num_classes 34 \
#	                                 --class_cond True --no_instance False --training_step 100000 --use_vae True \
#	                                  --catch_path /home/harry/Data_generation/train270 #--resume_checkpoint ../OUTPUT/Cityscapes270-SDM-256CH-500epoch/model080000.pt \

# Classifi er-free Finetune
#export OPENAI_LOGDIR='../OUTPUT/Cityscapes-SDM-256CH-10epoch-FINETUNE'
#mpiexec --allow-run-as-root -np 4 python ../image_train.py --data_dir /data1/dataset/Cityscapes --dataset_mode cityscapes --lr 2e-5 --batch_size 4 --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 256 --learn_sigma True \
#	     --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --use_checkpoint True --num_classes 34 \
#	     --class_cond True --no_instance False --drop_rate 0.2 --resume_checkpoint ../OUTPUT/Cityscapes-SDM-256CH-10epoch/model002000.pt --training_step 2000



# Testing
export OPENAI_LOGDIR='../OUTPUT/Cityscapes270-VAESDM-128CH-30k-TEST'
mpiexec -np 2 python3 ../image_sample.py --data_dir /media/rick/f7a9be3d-25cd-45d6-b503-7cb8bd32dbd5/cityscape_synthetic --dataset_mode cityscapes --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 270 --learn_sigma True \
       --noise_schedule linear --num_channels 128 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --num_classes 34 \
       --class_cond True --no_instance False --batch_size 4 --num_samples 4 --model_path ../OUTPUT/Cityscapes270-VAESDM-128CH-500epoch/model030000.pt --results_path ../RESULTS/Cityscapes270-VAESDM-128CH-30k --s 1.5 --use_vae True