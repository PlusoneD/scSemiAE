# scSemiAE
The implementation of "scSemiAE: A deep model with semi-supervised learning for single-cell transcriptomics". Single cell Semi-supervised AutoEncoder is a dimensionality reduction approach for better identification of cell subpopulations. For reproducing the paper results, please visit ***.



## Install

pip install git+https://github.com/PlusoneD/scSemiAE.git



## About file

│  README.md    // help
└─code                // related codes of scSemiAE
    │  data.py        //  data simulation for test
    │  run.py          //  an example
    │  dataset 
          │   expression_matrix.txt       
          └─  metadata.txt          
    └─model
          │   dataset.py        // dataset type for training model 
          │    inference.py    // classification method
          │   loss.py              //  loss function
          │    metrics.py       //  metrics calculation 
          │   net.py                //  model network 
          │   scSemiAE.py     //  model class
          └─  utils.py            //  some tool function



## Usage

To train a model based on a dataset, please run 'run.py'. The parameters are listed below.



--data_path: path to the dataset folder, default="./dataset/"

--save_path: path to the output directory, default="./output/"

--lab_size: labeled set size for each cell type, default=10

--lab_ratio: labeled set ratio for each cell type, default= -1

--cuda: enables cuda

--pretrain_batch: batch size for pretraining, default=100

--epochs: number of epochs to train for, default=60

--nepoch_pretrain: number of epochs to pretrain for, default=50

--learning_rate: learning rate for the model, default=0.001

--lr_scheduler_step: StepLR learning rate scheduler step, default=10

--lr_scheduler_gamma: StepLR learning rate scheduler gamma, default=0.5

--Lambda: weight for L2, default=1

--visual: visualization of data. default=False



if you don't want to change the parameter, you can only put the data (two files) into "./dataset/" and execute: 

python run.py

of note，the data need to meet some requirements such as "expression_matrix.txt" and "metadata.txt".
