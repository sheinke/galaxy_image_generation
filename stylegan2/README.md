# StyleGAN2

Modified from https://github.com/manicman1999/StyleGAN2-Tensorflow-2.0  
Licence of code is MIT


## Usage
### training
`python stylegan2_v2.py --run_name MY_MODEL --data_root PATH_TO_DATA`  
*run_name* is a model version, it can be reused only 
if the results of the previous run with the same *run_name* are moved or deleted.
### evaluation
`python stylegan2_v2.py`  
`--run_name EXAMPLE_EVALUATION`  
`--evaluate_mode  1`  
`--run_name_of_loaded_model TRAINED_MODEL_RUN_NAME`  
`--number_of_loaded_model NR_OF_CHECKPOINT`  
`--nr_eval_images NR_OF_IMAGES_T0_GENERATE`  
### transfer learning
`python stylegan2_v2.py`  
`--run_name RETRAINING`  
`--data_root PATH_TO_DATA`   
`--run_name_of_loaded_model BASE_MODEL`  
`--number_of_loaded_model NR_OF_CHECKPOINT`  

*note that if you would like to resume the training procedure approximately  where it was stopped please use*
 `--set_steps 1` *; by default the training procedure will load the specified model 
 and intialize a new training procedure*

#### local  run
if you run your code not on the Leonhard cluster set `--use_scratch 0`  
if you would like to debug locally with small amount of data you can run
`python stylegan2_v2.py --run_name MY_MODEL --data_root PATH_TO_DATA --debug_mode 1`

#### parameters
please check the list of other arguments
`python stylegan2_v2.py --help`

#### visualisations
if you would like to monitor losses and generated images in real time,
open a wandb *cil_stylegan* project


---
### Example
train model on binarised images  
`python stylegan2_v2.py --run_name BINARISED --data_root PATH_TO_DATA --img_binarise 1`  

retrain on normal images the checkpointed model with checkpoint number 10  
`python stylegan2_v2.py --run_name RETRAINING -data_root PATH_TO_DATA 
--run_name_of_loaded_model BINARISED --number_of_loaded_model 10`


continue training on normal images where the training was stopped with checkpoint 5  
`python stylegan2_v2.py --run_name RETRAINING_CONTINUE -data_root PATH_TO_DATA 
--run_name_of_loaded_model RETRAINING --number_of_loaded_model 5 --set_steps 1`