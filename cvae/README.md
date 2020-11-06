# Convolutional Variational Autoencoder

This script defines a CVAE model and trains it on the given dataset. To start the training, call

`python main.py` 
`--data_root /path/to/theDataset/cosmology_aux_data_170429/cosmology_aux_data_170429/`   
`--output_dir /path/to/yourOutputDirectory`   

This will create three subfolders inside your output_dir. Tensorboard-readable statistics are stored in "/logs/", model training checkpoints are stored in "/checkpoints/" and results (i.e. images generated during training) are stored in "/results/".

Please note that we were unable to obtain usable cosmology images with this script, despite extensive experiments.
