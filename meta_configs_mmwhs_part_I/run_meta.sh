# Run main script for every meta-configuration file

main_script_path=../main_slice_inflate.py

for meta_config in *.json
do
    echo "Running $meta_config"
    python $main_script_path --meta_config_path $meta_config
done