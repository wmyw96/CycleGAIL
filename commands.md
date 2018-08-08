# Commands to reproduce the experiment results

## Identity Mapping

### HalfCheetah Identity Mapping

- Baseline
```
python run_mujoco.py --name logs/halfcheetah-ident-baseline --lr 0.0001 --nhidd 128 --nhidf 128 --nhidg 128 --loss_metric L2 --epoch 100000 --ntraj 25 --lf 1.0 --lg 1.0 --loss wgan-gp --nd1 24 --nd2 24 --enva HalfCheetah-v1 --envb HalfCheetah-v1 --ckdir model --n_c 5 --gamma 10.0 --batch_size 100 --log_interval 200 --markov 0
```

- Markov Concat = 2
```
python run_mujoco.py --name logs/halfcheetah-ident-markov --lr 0.0001 --nhidd 128 --nhidf 128 --nhidg 128 --loss_metric L2 --epoch 100000 --ntraj 25 --lf 1.0 --lg 1.0 --loss wgan-gp --nd1 24 --nd2 24 --enva HalfCheetah-v1 --envb HalfCheetah-v1 --ckdir model --n_c 5 --gamma 10.0 --batch_size 100 --log_interval 200 --markov 2 --exp identity
```

### Walker2d Identity Mapping

- Baseline
```
python run_mujoco.py --name logs/walker2d-ident-baseline --lr 0.0001 --nhidd 128 --nhidf 128 --nhidg 128 --loss_metric L2 --epoch 100000 --ntraj 25 --lf 1.0 --lg 1.0 --loss wgan-gp --nd1 24 --nd2 24 --enva Walker2d-v1 --envb Walker2d-v1 --ckdir model --n_c 5 --gamma 10.0 --batch_size 100 --log_interval 200 --markov 0
```

- Markov Concat = 2
```
python run_mujoco.py --name logs/walker2d-ident-markov --lr 0.0001 --nhidd 128 --nhidf 128 --nhidg 128 --loss_metric L2 --epoch 100000 --ntraj 25 --lf 1.0 --lg 1.0 --loss wgan-gp --nd1 24 --nd2 24 --enva Walker2d-v1 --envb Walker2d-v1 --ckdir model --n_c 5 --gamma 10.0 --batch_size 100 --log_interval 200 --markov 2
```
