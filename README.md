### offline_advoran


# Install conda environment

Install and activate conda environment 

```
conda env create -f environment.yml
conda activate tf_agent_env

```

# Dataset

1. Download "rome_static_medium" dataset from [Dataset (https://github.com/wineslab/colosseum-oran-coloran-dataset.git)

2. Build dataset
```
python dataset_builder.py colosseum-oran-coloran-dataset/rome_static_medium dataset/embb_dataset.csv

```
 
# Train agent

```
python train.py

```

# Test agent

1. Change the policy directory by changing "policy_path"
```
def main():
    policy_path = 'drl_agent_files/run_20260209_092523/policy'
    print(f"--- Loading Policy from: {policy_path} ---")

```
2.  test agent
```
python test_agent.py

```

