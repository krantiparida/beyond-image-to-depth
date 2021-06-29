CUDA_VISIBLE_DEVICES=2 python train.py \
--validation_on \
--dataset mp3d \
--img_path /data1/kranti/audio-visual-depth/dataset/visual_echoes/images/mp3d_split_wise \
--metadatapath /data1/kranti/audio-visual-depth/dataset/visual_echoes/metadata/mp3d \
--audio_path /data1/kranti/audio-visual-depth/dataset/visual_echoes/echoes/mp3d/echoes_navigable \
--checkpoints_dir /data1/kranti/audio-visual-depth/checkpoints \
--init_material_weight ./checkpoints/material_pre_trained_minc.pth