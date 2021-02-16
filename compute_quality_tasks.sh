#!/bin/bash

CONFIG=/tmp3/peter0749/GDN_self/test_task_adapt.json
TASK_DATA_PREFIX=/tmp3/peter0749/EGAD/egad_val_tasks/task
TASK_LABEL_PREFIX=/tmp3/peter0749/EGAD/egad_val_tasks/task_label
TASK_ID=$1
WEIGHTS=/tmp3/peter0749/GDN_self/grasp_logs/gdn_self_train_egad_reptile_w_label_step_3_adam/best.ckpt
NSHOT=$2
NEPOCH=$3
BATCH_SIZE=$4
TRAIL_ID=$5
TASK_OUTPUT_PREFIX=/tmp3/peter0749/GDN_self/egad_outputs
JSON_OUTPUT=${TASK_OUTPUT_PREFIX}/${TASK_ID}_${NSHOT}_${NEPOCH}_${BATCH_SIZE}_${TRAIL_ID}.json

if [[ -f ${JSON_OUTPUT} ]]; then
    echo "File exists. Skipping $JSON_OUTPUT..."
    exit 0
fi

python compute_quality_scene.py ${CONFIG} ${TASK_OUTPUT_PREFIX}/${TASK_ID}_${NSHOT}_${NEPOCH}_${BATCH_SIZE}_${TRAIL_ID} ${JSON_OUTPUT} --pc_path ${TASK_DATA_PREFIX}/${TASK_ID} --workers 1
