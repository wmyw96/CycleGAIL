## Environments

### Set up a virtual environment

- using python version 3

### Install Mujoco

- Download `getid_osx` and change its permission using `chmod 755 [filename]`
- Apply for a student license using .edu mail
- put both `mjkey.txt` and `mujoco131` under `~/.mujoco/`

### Install Gym

- Firstly, install `mujoco-py` version 0.5.7 (since only 0.5.7 can incorporate with `mujoco 131`), and then install gym version 0.9.1 (newer version might have the problem when loading `Humanoid-v2`)

```
pip3 install mujoco-py==0.5.7
pip3 install mujoco==0.9.1
```

