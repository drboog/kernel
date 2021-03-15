Data folder is empty because of the limited space.

We provide some hyper-parameter settings here:

1. For improved SVGD, use the following command:

python trainer.py --dataset kin8nm --gamma_1 5 --gamma_2 5 --gamma_3 0.01 --kernel hk --learning_rate 0.005 --k_steps 5
dataset can be wine, protein, boston, kin8nm, combined, year, comcrete, 'n_epoches' are in the range of 100~1000

2. For improved SVGD with particle generator, use the followings:
(note that in model_svgd.py, the particle generator can be constructed with a neural network or learnable Gaussian,
 you may need to manually change it in model_svgd.py, leanable Gaussian is used for 'kin8nm' and 'concrete')

python trainer.py --imp 1 --dataset boston --gamma_1 10 --gamma_2 1 --gamma_3 0.0 --kernel hk --learning_rate 0.0002 --k_steps 10 --n_epoches 1000
python trainer.py --imp 1 --dataset combined --batch_size 1000 --gamma_1 10 --gamma_2 1 --gamma_3 0.0 --kernel hk --learning_rate 0.0002 --k_steps 10 --n_epoches 1000
python trainer.py --imp 1 --dataset concrete --gamma_1 10 --gamma_2 1 --gamma_3 0.0 --kernel hk --learning_rate 0.001 --k_steps 10 --n_epoches 1000
python trainer.py --imp 1 --dataset kin8nm --gamma_1 10 --gamma_2 1 --gamma_3 0.01 --kernel hk --learning_rate 0.0005 --k_steps 5 --n_epoches 1000
python trainer.py --imp 1 --dataset protein --gamma_1 10 --gamma_2 10 --gamma_3 0.0 --kernel hk --learning_rate 0.005 --k_steps 5 --n_epoches 100
python trainer.py --imp 1 --dataset wine --gamma_1 10 --gamma_2 1 --gamma_3 0.01 --kernel hk --learning_rate 0.001 --k_steps 5 --n_epoches 100
python trainer.py --imp 1 --dataset year --gamma_1 10 --gamma_2 1 --gamma_3 0.0 --kernel hk --learning_rate 0.0005 --k_steps 10 --n_epoches 100

3. For improved matrix-valued SVGD, just revise the '--method' to be 'svgd_kfac' or 'mixture_kfac' (see trainer.py)
For all datasets, learning rate is 0.001. 
k_steps is 10 for all datasets except 'kin8nm' with 'mixture_kfac' and 'boston' with 'svgd_kfac', 5 steps are used in these two.
'batch_size' are the same as previous experiments.
'n_epoches':  boston(200), combined(1000), concrete(800 for svgd_kfac, 1000 for mixture_kfac)
                   kin8nm(500 for svgd_kfac, 200 for mixture_kfac), protein(100), wine(100), year(100)
'gamma_1' and 'gamma_2' are selected from (5, 5) (10, 1), (10, 10), try --gamma_1 10 --gamma_2 1 first

