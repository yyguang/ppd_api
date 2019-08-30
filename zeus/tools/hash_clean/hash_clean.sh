DATASET_LIST_PATH='./other.txt'
SIM_FAMS_SAVE_PATH='./tmp/'


MODE='clean'
HASH_TYPE='phash'
IS_LOAD_HASH_LIST=1
IS_SAVE_HASH_LIST=0
HAMMING_DIST_THR=5


LIST_FILE=${DATASET_LIST_PATH##*/}
LOG_PATH=${MODE}.${LIST_FILE%.*}.${HASH_TYPE}.${HAMMING_DIST_THR}.log

nohup python -u hash_clean.py \
      --mode ${MODE} \
      --hash_type ${HASH_TYPE} \
      --dataset_list_path ${DATASET_LIST_PATH} \
      --is_load_hash_list ${IS_LOAD_HASH_LIST} \
      --is_save_hash_list ${IS_SAVE_HASH_LIST} \
      --hamming_dist_thr ${HAMMING_DIST_THR} \
      --sim_fams_save_path ${SIM_FAMS_SAVE_PATH} \
      >${LOG_PATH} 2>&1 &
