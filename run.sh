#!/bin/bash
# -------------------------------------------------------------------------------
# This shell script fine-tunes the main condition attn python script
# -------------------------------------------------------------------------------

# Quit if there's any errors
set -e

CUDA_ID=1
DATASET="spouse"
SEED=0
EPOCH=500

for LR in 0.02
do
  for HIDDEN in 256
  do
    for C2 in 0.2 # c1 in the paper
    do
      for C3 in 0.1
        do
          for N_HIGH_COV in 1
          do
            CUDA_VISIBLE_DEVICES="$CUDA_ID" python main_conditional_attn.py \
                                            --ds "$DATASET" \
                                            --seed "$SEED" \
                                            --epoch "$EPOCH" \
                                            --lr "$LR" \
                                            --hidden "$HIDDEN" \
                                            --c2 "$C2" \
                                            --c3 "$C3" \
                                            --n_high_cov "$N_HIGH_COV"
          done
        done
      done
    done
  done
done


