# Run main script for every meta-configuration file
cd ..
main_script_path=./main_slice_inflate.py

for meta_config in ./meta_configs_mmwhs_part_I/*.json
do
    echo "Running $meta_config"
    python $main_script_path --meta_config_path $meta_config >> ./meta_configs_mmwhs_part_I/log_all.txt
done