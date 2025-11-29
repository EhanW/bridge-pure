# BS=64


DATASET_NAME=$1
PRED=$2
ATTACK=$3
NUM_TRAINING_DATA=$4
NUM_PURIFYING_DATA=$5

NGPU=1
BS=256

SIGMA_MAX=80.0
SIGMA_MIN=0.002
SIGMA_DATA=0.5
COV_XY=0


NUM_CH=256
ATTN=32,16,8
SAMPLER=real-uniform
NUM_RES_BLOCKS=2
USE_16FP=True
ATTN_TYPE=flash



IMAGE_DIR="."

if [[ $DATASET_NAME == "cifar10" ]]; then
    DATA_DIR="${IMAGE_DIR}/images/${DATASET_NAME}/${ATTACK}/${NUM_TRAINING_DATA}_${NUM_PURIFYING_DATA}"
    DATASET=cifar10
    IMG_SIZE=32
    NUM_CH=192
    NUM_RES_BLOCKS=3
    EXP="${DATASET_NAME}_${ATTACK}_${NUM_TRAINING_DATA}_${NUM_PURIFYING_DATA}_${IMG_SIZE}_${NUM_CH}d"
    SAVE_ITER=100000

elif [[ $DATASET_NAME == "cifar100" ]]; then
    DATA_DIR="${IMAGE_DIR}/images/${DATASET_NAME}/${ATTACK}/${NUM_TRAINING_DATA}_${NUM_PURIFYING_DATA}"
    DATASET=cifar100
    IMG_SIZE=32
    NUM_CH=192
    NUM_RES_BLOCKS=3
    EXP="${DATASET_NAME}_${ATTACK}_${NUM_TRAINING_DATA}_${NUM_PURIFYING_DATA}_${IMG_SIZE}_${NUM_CH}d"
    SAVE_ITER=100000
    
elif [[ $DATASET_NAME == "webfacesubset" ]]; then
    DATA_DIR="${IMAGE_DIR}/images/${DATASET_NAME}/${ATTACK}/${NUM_TRAINING_DATA}_${NUM_PURIFYING_DATA}"
    DATASET=webfacesubset
    BS=32  # was set 64 on tml4 a100. but for GPU memory, set  now
    IMG_SIZE=112
    NUM_CH=192
    NUM_RES_BLOCKS=3
    EXP="${DATASET_NAME}_${ATTACK}_${NUM_TRAINING_DATA}_${NUM_PURIFYING_DATA}_${IMG_SIZE}_${NUM_CH}d"
    SAVE_ITER=5000 # 20000 # 100000
    TOTAL_TRAINING_STEPS=10000 # 80000 # 100000

elif [[ $DATASET_NAME == "imagenetsubset" ]]; then
    DATA_DIR="${IMAGE_DIR}/images/${DATASET_NAME}/${ATTACK}/${NUM_TRAINING_DATA}_${NUM_PURIFYING_DATA}"
    DATASET=imagenetsubset
    BS=16
    IMG_SIZE=224
    NUM_CH=192
    NUM_RES_BLOCKS=3
    EXP="${DATASET_NAME}_${ATTACK}_${NUM_TRAINING_DATA}_${NUM_PURIFYING_DATA}_${IMG_SIZE}_${NUM_CH}d"
    SAVE_ITER=100000
    TOTAL_TRAINING_STEPS=100000
fi
    
if  [[ $PRED == "ve" ]]; then
    EXP+="_ve"
    COND=concat
elif  [[ $PRED == "vp" ]]; then
    EXP+="_vp"
    COND=concat
    BETA_D=2
    BETA_MIN=0.1
    SIGMA_MAX=1
    SIGMA_MIN=0.0001
elif  [[ $PRED == "ve_simple" ]]; then
    EXP+="_ve_simple"
    COND=concat
elif  [[ $PRED == "vp_simple" ]]; then
    EXP+="_vp_simple"
    COND=concat
    BETA_D=2
    BETA_MIN=0.1
    SIGMA_MAX=1
    SIGMA_MIN=0.0001
else
    echo "Not supported"
    exit 1
fi



# if [[ $IMG_SIZE == 256 ]]; then
#     BS=16
# else
#     BS=192
# fi

