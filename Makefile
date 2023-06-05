download_pretrained_weights:
	mkdir -p checkpoints/s2ef_efwt/all/schnet
	wget https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/s2ef/schnet_all_large.pt -O checkpoints/s2ef_efwt/all/schnet/schnet_all_large.pt

download_datasets:
	wget https://figshare.com/ndownloader/files/40999481?private_link=18ed6523b3dd6aeb2bcb -O data/cuformate.tgz
	tar -xvzf data/cuformate.tgz -C data

prepare_experiments: download_pretrained_weights download_datasets

run_exp:
