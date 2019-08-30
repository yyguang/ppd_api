DATASET_LIST_PATH='/data/project/porn20180327/training_daily-180327.txt'
SRC_IMG_PATH='./511ff846a0d54be605ef90115ea9c1e7.jpg'
SIM_IMGS_SAVE_PATH='0'


MODE='find'
HASH_TYPE='phash'
IS_LOAD_HASH_LIST=1
IS_SAVE_HASH_LIST=0
HAMMING_DIST_THR=10


LIST_FILE=${DATASET_LIST_PATH##*/}
LOG_PATH=${MODE}.${LIST_FILE%.*}.${HASH_TYPE}.${HAMMING_DIST_THR}.log

nohup python -u hash_clean.py \
      --mode ${MODE} \
      --hash_type ${HASH_TYPE} \
      --dataset_list_path ${DATASET_LIST_PATH} \
      --is_load_hash_list ${IS_LOAD_HASH_LIST} \
      --is_save_hash_list ${IS_SAVE_HASH_LIST} \
      --hamming_dist_thr ${HAMMING_DIST_THR} \
      --src_img_path ${SRC_IMG_PATH} \
      --sim_imgs_save_path ${SIM_IMGS_SAVE_PATH} \
      >${LOG_PATH} 2>&1 &
