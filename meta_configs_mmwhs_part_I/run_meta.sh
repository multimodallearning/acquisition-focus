#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/..
# Run main script for every meta-configuration file
main_script_path=./main_slice_inflate.py

for meta_config in $SCRIPT_DIR/*.json
do
    # Create a new log file for each meta-configuration file
    log_file_name=$(basename $meta_config .json).log

    echo "Using meta-config $meta_config"
    python $main_script_path --meta_config_path $meta_config >> $SCRIPT_DIR/$log_file_name
done