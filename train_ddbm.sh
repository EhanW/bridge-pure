DATASET_NAME=$1
PRED=$2
ATTACK=$3
NUM_TRAINING_DATA=$4
NUM_PURIFYING_DATA=$5

CKPT=$6

FREQ_SAVE_ITER=500
NGPU=1
TOTAL_TRAINING_STEPS=100000

source ./args.sh $DATASET_NAME $PRED $ATTACK $NUM_TRAINING_DATA $NUM_PURIFYING_DATA


mpiexec -n $NGPU python scripts/ddbm_train.py --exp=$EXP \
 --attention_resolutions $ATTN --class_cond False --use_scale_shift_norm True \
  --dropout 0.1 --ema_rate 0.9999 --batch_size $BS \
   --image_size $IMG_SIZE --lr 0.0001 --num_channels $NUM_CH --num_head_channels 64 \
    --num_res_blocks $NUM_RES_BLOCKS --resblock_updown True ${COND:+ --condition_mode="${COND}"} ${MICRO:+ --microbatch="${MICRO}"} \
     --pred_mode=$PRED  --schedule_sampler $SAMPLER ${UNET:+ --unet_type="${UNET}"} \
    --use_fp16 $USE_16FP --attention_type $ATTN_TYPE --weight_decay 0.0 --weight_schedule bridge_karras \
     ${BETA_D:+ --beta_d="${BETA_D}"} ${BETA_MIN:+ --beta_min="${BETA_MIN}"}  \
      --data_dir=$DATA_DIR --dataset=$DATASET ${CH_MULT:+ --channel_mult="${CH_MULT}"} \
      --num_workers=8  --sigma_data $SIGMA_DATA --sigma_max=$SIGMA_MAX --sigma_min=$SIGMA_MIN --cov_xy $COV_XY \
      --save_interval_for_preemption=$FREQ_SAVE_ITER --save_interval=$SAVE_ITER --debug=False \
      --total_training_steps=$TOTAL_TRAINING_STEPS \
      ${CKPT:+ --resume_checkpoint="${CKPT}"} 