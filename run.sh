#!/bin/bash
# -------------------------------------------------------------------------------
# This shell script fine-tunes the attention 6.1 python script
# -------------------------------------------------------------------------------

# Quit if there's any errors
set -e

CUDA_ID=1
DATASET="imdb"
SEED=0
EPOCH=500

for LR in 0.02
do
  for HIDDEN in 128
  do
    for C2 in 0.15
    do
      for C3 in 0.15
      do
        for C4 in 0.15
        do
          for N_HIGH_COV in 2
          do
            CUDA_VISIBLE_DEVICES="$CUDA_ID" python main_conditional_attn.py \
                                            --ds "$DATASET" \
                                            --seed "$SEED" \
                                            --epoch "$EPOCH" \
                                            --lr "$LR" \
                                            --hidden "$HIDDEN" \
                                            --c2 "$C2" \
                                            --c3 "$C3" \
                                            --c4 "$C4" \
                                            --n_high_cov "$N_HIGH_COV"
          done
        done
      done
    done
  done
done


