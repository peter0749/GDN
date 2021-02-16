#!/bin/bash

CONFIG=/tmp3/peter0749/GDN_self/test_task_adapt.json
TASK_DATA_PREFIX=/tmp3/peter0749/EGAD/egad_val_tasks/task
TASK_LABEL_PREFIX=/tmp3/peter0749/EGAD/egad_val_tasks/task_label
TASK_ID=$1
WEIGHTS=/tmp3/peter0749/GDN_self/reptile.ckpt
NSHOT=$2
NEPOCH=50
BATCH_SIZE=$4
TRAIL_ID=$5
TASK_OUTPUT_PREFIX=/tmp3/peter0749/GDN_self/egad_outputs_50_epochs

python eval_reptile.py ${CONFIG} ${TASK_DATA_PREFIX}/${TASK_ID} ${TASK_LABEL_PREFIX}/${TASK_ID} ${WEIGHTS} ${NSHOT} ${NEPOCH} ${BATCH_SIZE} ${TASK_OUTPUT_PREFIX}/${TASK_ID}_${NSHOT}_${NEPOCH}_${BATCH_SIZE}_${TRAIL_ID}

