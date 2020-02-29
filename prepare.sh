# With regular sampling on k-space
#python generate_data.py --yaml_path=config/construct_data_regular.yaml
#python split_data.py --yaml_path=config/split_data_regular.yaml

# With uniform sampling on k-space
python generate_data.py --yaml_path=config/construct_data_uniform.yaml
python split_data.py --yaml_path=config/split_data_uniform.yaml
