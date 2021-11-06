# MCRL
Solving Routing Problems through Monte Carlo Tree Search-Based Training Pipeline <br>
## To train the model (e.g. TSP with 20 nodes)
```
python run.py --graph_size 20 --baseline rollout --run_name 'tsp20' --problem="tsp" --batch_size=512 --epoch_size=1280000 --kl_loss=0.01 --n_EG=2 --n_paths=5 --val_size=10000
```
## To generate the test dataset for TSP20
```
mkdir data
python generate_data.py --filename=data/tsp_20 --problem="tsp" --graph_sizes=20 --dataset_size=100000
```
## To evaluate the test dataset with pretrained model for TSP20
```
CUDA_VISIBLE_DEVICES="0" python -u search.py data/tsp_20.pkl --model=pretrained/tsp_20/epoch-99.pt --beam_size=1 --eval_batch_size=1024
```
## Dependencies<br>
Python>=3.6 <br>
tqdm>=4.36.1 <br>
torch==1.4.0 <br>
tensorboardX>=1.9 <br>
numpy>=1.18.1 <br>
cvxpy==1.1.0a1 <br>
scipy==1.4.1<br>
# Testing
```
python eval.py data/tsp/tsp20_test_seed1234.pkl --model pretrained/tsp_20 --decode_strategy 
```
