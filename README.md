# CycleGAIL

## Requirements

- Environment: mujoco131, mujoco-py==0.5.7, gym==0.9.1
- Models: python==2.7, tensorflow, numpy, matplotlib

## Run

Commands to reproduce:

```
python run_mujoco.py --name logs/halfcheetah-ident --lr 0.0001 --nhidd 128 --nhidf 128 --nhidg 128 --loss_metric L2 --epoch 100000 --ntraj 100 --lf 1.0 --lg 1.0 --loss wgan-gp --nd1 90 --nd2 90 --enva HalfCheetah-v1 --envb HalfCheetah-v1 --ckdir model --n_c 5 --mode test
```
