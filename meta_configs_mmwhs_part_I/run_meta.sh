# Run main script for every meta-configuration file

main_script_path=python ../main_slice_inflate.py

for meta_config in *.json
do
    echo "Running $config"
    $main_script_path $config
done