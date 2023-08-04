#!/bin/zsh
screen -L -S carla_exp .venv/bin/python
python3 code/search_hyperparameters.py