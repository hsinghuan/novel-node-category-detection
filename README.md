# novel-node-category-detection

Experimental code for "Novel Node Category Detection Under Subpopulation Shift".

## Dataset Preparation
Please use the scripts in `data_preprocess/` for dataset preprocessing.
Use `planetoid.py` for preprocessing Cora and CiteSeer, `amazon.py` for Computers and Photo, and `arxiv.py` for arxiv.

Example usages:
```
python planetoid.py --root_dir /path/to/data/rootdir/ --dataset Cora --preprocess_name shift
```

```
python arxiv.py --root_dir /path/to/data/rootdir/
```

## Experiment Execution
In `config/config.yaml`, enter the name of the detector yaml file corresponding to a method in the `defaults:detector:` entry.
Then, uncomment the configurations corresponding to the dataset you intend to test on.
Note that the configurations might be different for each dataset so please uncomment the lines in the correct block.

For RECO-SLIP, use --multirun over multiple target recalls.

Example usages:
### Train RECO-SLIP
```
python train.py --multirun target_recalls=[0.05],[0.10],[0.15],[0.20],[0.25] seed=10
```

### Test RECO-SLIP
```
python test.py seed=10
```