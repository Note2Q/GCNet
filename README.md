# My Paper Title

This repository is the implementation of **Generative Causality-Guided Network Toward Structured Multi-Task Learning**

## Requirements
Below is environment built with pytorch-geometric.
To install requirements:

```
conda install -y -c pytorch pytorch=1.6.0 torchvision
conda install -y matplotlib
conda install -y scikit-learn
conda install -y -c rdkit rdkit=2019.03.1.0
conda install -y -c anaconda beautifulsoup4
conda install -y -c anaconda lxml

wget https://data.pyg.org/whl/torch-1.6.0%2Bcu102/torch_sparse-0.6.9-cp37-cp37m-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.6.0%2Bcu102/torch_scatter-2.0.6-cp37-cp37m-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.6.0%2Bcu102/torch_spline_conv-1.2.1-cp37-cp37m-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.6.0%2Bcu102/torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl
pip install torch_sparse-0.6.9-cp37-cp37m-linux_x86_64.whl
pip install torch_scatter-2.0.6-cp37-cp37m-linux_x86_64.whl
pip install torch_spline_conv-1.2.1-cp37-cp37m-linux_x86_64.whl
pip install torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl
pip install torch-geometric==1.6.*
```


## Training

To train the model(s) in the paper, we provide the scripts for training. All the optimal hyper-parameters are provided in the bash scripts.

```train
bash feature_space.sh
bash output_space.sh
```

## Evaluation

To evaluate the model, run:

```eval
cd checkpoint
bash eval_GCNet.sh > eval_GCNet.out
```



## Baselines
We also provide the scripts for all STL and MTL baselines under the scripts folder.