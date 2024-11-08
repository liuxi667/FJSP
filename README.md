# FJSP

This repository is the implementation of the paper “[Flexible Job Shop Scheduling Problem Using Graph Neural Networks and Reinforcement Learning]

### requirements

- python $=$ 3.8.19
- opencv-python $=$ 4.9.0.80
- numpy $=$ 1.24.1
- ortools $=$ 9.9.3963
- pandas $=$ 2.0.3
- torch $=$ 2.2.2+cu121
- torchaudio $=$ 2.2.2+cu121
- torchvision $=$ 0.17.2+cu121
- tqdm $=$ 4.66.2

### introduction

- `data` saves the instance files including testing instances (in the subfolder `BenchData`, `SD1` and `SD2`) and validation instances (in the subfolder `data_train_vali`) .
- `model` contains the implementation of the proposed framework.
- `or_solution` saves the results solved by Google OR-Tools.
- `test_results`saves the results solved by priority dispatching rules and DRL models.
- `train_log` saves the training log of models, including information of the reward and validation makespan.
- `trained_network` saves the trained models.
- `common_utils.py` contains some useful functions (including the implementation of priority dispatching rules mentioned in the paper) .
- `data_utils.py` is used for data generation, reading and format conversion.
- `fjsp_env_same_op_nums.py` and `fjsp_env_various_op_nums.py` are implementations of fjsp environments, describing fjsp instances with the same number of operations and different number of operations, respectively.
- `ortools_solver.py` is used for solving the instances by Google OR-Tools.
- `params.py` defines parameters settings.
- `print_test_result.py` is used for printing the experimental results into an Excel file.
- `test_heuristic.py` is used for solving the instances by priority dispatching rules.
- `test_trained_model.py` is used for evaluating the models.
- `train.py` is used for training.

### train

```python
python train.py # train the model on 10x5 FJSP instances using SD1

# options (Validation instances of corresponding size should be prepared in ./data/data_train_vali/{data_source})
python train.py 	--n_j 10		# number of jobs for training/validation instances
			--n_m 5			# number of machines for training/validation instances
    			--data_source SD2	# data source (SD1 / SD2)
        		--data_suffix mix	# mode for SD2 data generation
            					# 'mix' is thedefault mode as defined in the paper
                				# 'nf' means 'no flexibility' (generating JSP data) 
        		--model_suffix demo	# annotations for the model
```

### evaluate

```python
python test_trained_model.py # evaluate the model trained on '10x5+mix' of SD1 using the testing instances of the same size using the greedy strategy

# options (Model files should be prepared in ./trained_network/{model_source})
python test_trained_model.py 	--data_source SD1	# source of testing instances
				--model_source SD1	# source of instances that the model trained on 
    				--test_data 10x5+mix	# list of instance names for testing
        			--test_model 10x5+mix	# list of model names for testing
            			--test_mode False	# whether using the sampling strategy
                		--sample_times 100	# set the number of sampling times
```


## Reference
- https://github.com/wrqccc/FJSP-DRL
- https://github.com/songwenas12/fjsp-drl/
- https://github.com/zcaicaros/L2D
- https://github.com/google/or-tools
- https://github.com/Diego999/pyGAT

