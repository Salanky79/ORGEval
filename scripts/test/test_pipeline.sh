#!/bin/bash

set -x 

export OPENAI_API_KEY="your api key"
export OPENAI_BASE_URL="your api base url"
MODEL_NAME="gpt-4o"


# Default values for arguments
DATA_DIR="data/bench4opt"
SAMPLE_ID=""  # default is empty, meaning all samples will be processed
FILE_NAME="test.jsonl"  # default file name for test data
OUTPUT_FILENAME=""  # output file name, if empty, the default name will be used

# Parse arguments
while getopts "d:s:f:o:" opt; do
    case $opt in
        d) DATA_DIR="$OPTARG";;
        s) SAMPLE_ID="--sample_id $OPTARG";;
        f) FILE_NAME="$OPTARG";;
        o) OUTPUT_FILENAME="$OPTARG";;
    esac
done

# Use a fixed file name to support resuming from a checkpoint
if [ -z "$OUTPUT_FILENAME" ]; then
    FIXED_FILENAME="test_pipeline_model_${MODEL_NAME//\//-}_resume.json"
    LOG_FILENAME="test_pipeline_model_${MODEL_NAME//\//-}_resume.log"
else
    FIXED_FILENAME="$OUTPUT_FILENAME"
    LOG_FILENAME="${OUTPUT_FILENAME%.json}.log"
fi

mkdir -p outputs

# Use the script that supports resuming from a checkpoint
python -m tests.test_full_pipeline_resume \
    --model_name $MODEL_NAME \
    --save_path outputs/$FIXED_FILENAME \
    --save_interval 1 \
    --data_dir $DATA_DIR \
    2>&1 | tee outputs/$LOG_FILENAME
