project/
│
├── main.py                      	# Entry point (runs pipeline)
├── config.py                    	# Config class + CLI parser
│
├── data/
│   └── loader.py                	# Load dataframe + select features/target
│
├── preprocessing/
│   ├── filters.py               	# Physics-based filtering (RiB, H, ranges)
│   ├── target_transform.py      	# log(H+1), log(tau)+1, inverse transform
│   └── split_scale.py           	# train/test split + scaling (ANN)
│
├── models/
│   ├── ann_model.py             	# ANN wrapper (train/predict/save/load)
│   └── mlp_wrapper.py           	# Wrapper for Bayesian tuning (skopt)
│   ├── rf_model.py             	# RF wrapper (train/predict/save/load)
│   └── xgb_wrapper.py           	# XGBoost wrapper (train/predict/save/load)
├── tuning/
│   └── ann_tune.py        		# Phase-wisetuning
│   └── rf_tune.py
|   └── xgb_tune.py
├── utils/
│   ├── load_hyperparams.py         	# Load saved hyperparameters
│   ├── experiment_logger.py       	# Track every run
│   └── plotting.py                   	# Plotting obs vs pred
││
├── outputs/                		# All Ensemble outputs go here
│   ├── final_summary*.csv     	  	# Ensemble mean metrics 
│   ├── ensemble_preds/            	# Ensemble averaged predictions
│   ├── seed_metrics/                 	# Ensemble seed metrics in .csv files
│   └── *plots/				# Store the ensemble plots
│
├── saved_models/                	# All outputs go here
│   ├── params/{folder}/*.pkl    	# trained models
│   ├── plots/{folder}/*.svg    	# plots of predicting ensemble ans single runs
│   ├── stats/				# Normalisation stats for feature scaling
│   ├── Hyperparameters/
|	├── *_params.json 		# tuned hyperparameters               
│   	├── *_conv.svg               	# convergence plots
│   	└── *_obj.svg                	# objective plots
│	└── *_cv.csv			#CV_results of tuning
│   ├── experiment_log.csv              # Tracker every run of main
|
├── hpt_tuning.ipynb			# notebook to carry out hyperparameter tuning
├── Training.ipynb			# To train single/ensemble case
├── Results.ipynb			# Plot ensemble results
├── Extract_params.ipynb		# Make eqn of final model parameters
├── Final_eqn_ANN.txt			# Final equations for ANN'
└── demo_inference.ipynb		# To run inference on random/unseen data.

