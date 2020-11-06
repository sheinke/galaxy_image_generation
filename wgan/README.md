# Gradient-penalized Wasserstein GAN

## Generative Training
To start the training of the WGAN, call

`python main.py`   
`--data_root /path/to/theDataset/cosmology_aux_data_170429/cosmology_aux_data_170429/`   
`--output_dir /path/to/yourOutputDirectory`   

This will create three subfolders inside your output_dir. Tensorboard-readable statistics are stored in "/logs/", model training checkpoints are stored in "/checkpoints/" and results (i.e. images generated during training) are stored in "/results/".


## Image Generation.
In order to obtain arbitrary amounts of generated images, use the generate_images.py script:

`python generate_images.py`   
`--n_images 42424242`   
`--generator_path /path/to/yourTrainedGenerator.h5`   
`--output_dir /path/to/yourOutputDirectory`   

This will create n_images images inside output_dir. Depending on the number of images to generate, this may take some time.
The script expects an H5 file, so you may need to convert the Tensorflow checkpoints that are created during training into this format (see next section). Note that, by default, the training script performs said conversion after all epochs have been completed.


## Checkpoint Conversion.
To convert the latest checkpoint of your training into an H5 file (maybe because you interrupted an ongoing training run), call

`python manager_checkpoint_to_h5.py`   
`--checkpoint_path /path/to/yourPreviouslyUsedCheckpointDirectory`   

This will create an H5 file for each of the trained generator and the discriminator inside checkpoint_path.
