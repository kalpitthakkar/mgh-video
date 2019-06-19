#!/usr/bin/env/bash
python shard_creation.py \
    --frames_per_clip 16 \
    --phase mgh_test \
    --num_shards 2 \
    --frame_height 256 \
    --frame_width 256 \
    --channels 3 \
    --multithread 0 \
    --transition 0 \
    --clips_per_behavior 70000 \
    "$@"
