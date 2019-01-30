# Tic Toe RL Agent
Tic Tac Toe playing RL agent in Python

## Installing the dependency
- Activate virtualenv (optional)
- `pip install -r requirements.txt`

## Training the agent

### To train the agent
`python3 run.py train`
model will be created at default path

### Provide model save path
`python3 run.py train --save=10k_episodes`
model will be saved inside models folder with provided name using pickle dump

## Playing against agent
`python3 run.py play`

### Provide model for playing
`python3 run.py play --model=10k_episodes`

### Check additional options with
`python3 run.py train --help` or
`python3 run.py play --help`

