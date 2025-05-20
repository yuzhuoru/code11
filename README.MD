Open source code of D2AC

Instruction

Command:
python d2ac_main.py \
--seed 1 \
--env_id Walker2d-v4 \
--delay_mode obs \
--obs_delay_dis gamma

delay_mode option: obs/act/both \
delay_dis option: gamma/uniform/doublegaussian \
You can add or modify delay distributions in utils/delay_distribution.py

Environment requirements: \
python==3.8.12 \
gym==0.26.2 \
gym-notices==0.0.8 \
gymnasium==0.29.0 \
imageio==2.34.2 \
mujoco==2.2.0 \
torch==1.12.0 
