# Training Neural Networks with Mixed Integer Programming

To run experiments:

python src/run_script.py --exp=exp1
python src/run_script.py --exp=exp2
python src/run_script.py --exp=exp3
python src/run_script.py --exp=exp4

To run any model:

python main_run.py --time=36000 --bound=1 --data=adult --seed=99413 --hl=16 --ex=100 --loss=min_hinge

time: time limit of model
bound: P value for limit of network values
data: dataset to use (mnist, adult, heart)
seed: specify seed for randomness
hl: number of nodes in hidden layers (16-16 for 2 layers with 16 nodes f.ex.)
ex: number of examples to run on
loss: model to use (sat_margin,min_hinge,max_correct,gd_nn,min_w,max_m)