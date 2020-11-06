# Score Prediction

## Kaggle submission

To reproduce our Kaggle submission using the ensemble of five finetuned WGAN critics, please run:
`bash predict_ensemble.sh`

We assume the following image directory: `./cil_data/cosmology_aux_data_170429/cosmology_aux_data_170429/query`.
To change the data directory, please modify the `img_dir` parameter.

The script uses 5 trained models from the `ensemble_models` folder.
In case you are using the submitted zipped version of the repository, make sure to download the binaries from our Gitlab-repository (you can find the link in the report PDF).

To train such models from scratch, please use our WGAN finetuning script.
Please change then `score_pred_mean` and `score_pred_std` parameters to 
values specific to your finetuned model (you can find them in the finetuning script output).

## WGAN finetuning
This script takes a pretrained model, adds new layers after a specified layer in the old model and finetunes the freshly added layers for score prediction (while keeping the pretrained / old layers frozen).
When the val error does not decrease anymore, it unfreezes the old layers and trains them jointly. It then restores the checkpoint that performed best on the val set and produces a final query submission csv file.

Assuming that "yourPretrainedDiscriminator.h5" contains a pretrained model and that "intermediateLayer" is the name of the layer after which you would like to add new layers, you would call the script as follows:

`python finetuning.py`   
`--model_file /path/to/yourPretrainedDiscriminator.h5`   
`--transfer_layer intermediateLayer`   
`--chkpnt_file finetunedScorePredictor`   
`--output_dir /path/to/yourOutputDirectory`   
`--data_root /path/to/theDataset/cosmology_aux_data_170429/cosmology_aux_data_170429/`   

Please note that "chkpnt_file" should simply be the desired name of your checkpoints (i.e. should not have a file-ending).

For instructions on how to pretrain a discriminator please refer to the `wgan` folder.


## Baselines
### Simple CNN
In order to obtain a baseline for accurate score prediction, we implemented a simple CNN. 
This script builds a simple CNN and trains it on fullsize images to accurately predict score values. It makes use of LR-reduction callbacks as well as early stopping model checkpointing.

To start the training, simply call

`python simple_cnn.py`   
`--output_dir /path/to/yourOutputDirectory`   
`--data_root /path/to/theDataset/cosmology_aux_data_170429/cosmology_aux_data_170429/`   

The output directory is used for storing the query result csv file, as well as model checkpoints.

### Feature-based baseline

Please find instructions in the `feature baseline` folder.

## Judge Script
To evaluate the quality of the images that our generative models produced, we wrote a simple "judge script" that loads the best predictor H5 binary and predicts scores for all PNG images inside a specified input directory.

Assuming that you call the script from this folder, simply do:

`python judge.py`   
`--img_dir /path/to/yourImagesWithManyPNGs/`   

This will output a csv file with the score predictions (Kaggle submission format), and print some basic statistics about the resulting score distribution (min, max, avg, std).
