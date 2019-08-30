# exp name and dataset
# model: inceptionV3/xception
exp_name="grindr_vline"
dataset="/data/project/data/grindr_vline_0815_dataset"
dense_units=1024
list_mode=1
se_block=0
model='inception_v3'

# parameters for multi-stage
# max batch_size for finetune added layers is 256 for 12GB Titan XP, 128 for 10 blocks
batch_size=256,128,128,128,128
# use ';' to seperate different stages for lr
lr="0.05;0.05;0.005;0.0005;0.0001" #"0.05;0.005;0.0005;0.0001"
epochs=16,16,16,16,16
# optimizer: sgd/adam/rmsprop
optimizer='sgd','sgd','sgd','sgd','sgd'
finetune_blocks=1,10,10,10,10
random_crop=0,0,0,0,0

validation_auc=0
test_auc=1
epoch_scale=1
cuda_devices="-1"

# resample on resize & letter box preprocess
interpolation='random'
letter_box=0

# early stopping
early_stopping=5

# optimizer
clipnorm=2.0

# lr decay
lr_decay=0
lr_decay_decayrate=0.94
lr_decay_decaystep=2

# data agumentation
# rotation: ref rotation_range in keras ImageDataGenerator
# rotation90: Whether turn on 0/90/180/270 realtime rotation variant
hflip=1
vflip=0
rotation=0
rotation90=0
# random crop
aspect_ratio_range=0.56,1.78
area_range=0.5,1.0
random_crop_list="md5_random_crop.list"
# color
brightness=0.1254902
saturation=0.5
# multiscale
zoom_pyramid_levels=4
zoom_short_side=160
# gaussian blur
gaussian_blur=0

# save data augmentation
save_to_dir="augmented_imgs"
save_prefix="aug"
save_samples_num=0

# advanced option
label_smoothing=0.0
weight_decay=0.00004
class_aware_sampling=0

# construct output dir name
run_time=`date +%Y%m%d%H%M`
new_name=${exp_name}_${run_time}_${USER}
output_dir=${new_name}

# backup source files
mkdir -p ${new_name}
# cp this script to output_dir
SCRIPT_NAME=$(basename $(readlink -f "$0"))
cp ${SCRIPT_NAME} ${output_dir}/
cp retrain_full.py ${output_dir}/
cp sm_sequence.py ${output_dir}/
cp evaluation.py ${output_dir}/
cp multi_modelcheckpoint_callback.py ${output_dir}/

# start running
nohup python -u retrain_full.py \
        --se_block ${se_block} \
        --list_mode ${list_mode} \
        --random_crop_list ${random_crop_list} \
        --class_aware_sampling ${class_aware_sampling} \
        --weight_decay ${weight_decay} \
        --label_smoothing ${label_smoothing} \
        --model ${model} \
        --interpolation ${interpolation} \
        --letter_box ${letter_box} \
        --dense_units ${dense_units} \
        --early_stopping ${early_stopping} \
        --finetune_blocks ${finetune_blocks} \
        --lr_decay ${lr_decay} \
        --lr_decay_decayrate ${lr_decay_decayrate} \
        --lr_decay_decaystep ${lr_decay_decaystep} \
        --optimizer ${optimizer} \
        --clipnorm ${clipnorm} \
        --validation_auc ${validation_auc} \
        --test_auc ${test_auc} \
        --learning_rate ${lr} \
        --epochs ${epochs} \
        --output_dir ${output_dir} \
        --cuda_devices ${cuda_devices} \
        --horizontal_flip ${hflip} \
        --vertical_flip ${vflip} \
        --rotation_range ${rotation} \
        --rotation90 ${rotation90} \
        --random_crop ${random_crop} \
        --aspect_ratio_range ${aspect_ratio_range} \
        --area_range ${area_range} \
        --zoom_pyramid_levels ${zoom_pyramid_levels} \
        --zoom_short_side ${zoom_short_side} \
        --brightness ${brightness} \
        --saturation ${saturation} \
        --gaussian_blur ${gaussian_blur} \
        --save_to_dir ${save_to_dir} \
        --save_prefix ${save_prefix} \
        --save_samples_num ${save_samples_num} \
        --epoch_scale ${epoch_scale} \
        --train_dir ${dataset}/train \
        --validation_dir ${dataset}/validation \
        --test_dir ${dataset}/test \
        --batch_size ${batch_size} >${new_name}/log 2>&1 &
