
# Full Pokemon States and Actions

## Pokemon
- type (up to 2)
- attack
- defense
- hp
- fast attack
- charged attack 1
- charged attack 2
- energy
### Fast attacks
- damage
- energy generated
- turns (2 turns is 1 second)
- type

### Charged attacks
- damage
- energy cost
- type
- charge (% damage you want to do)

## Player
- shields 
- number of pokemon remaining

## Global
- pokemon currently in play
---
# Phase 1

## Pokemon
- attack
- defense
- hp
- fast attack
- charged attack 1
- charged attack 2
- energy

### Fast attacks
- damage
- energy generated
- turns (2 turns is 1 second)

### Charged attacks
- damage
- energy cost

## Player
- 1 pokemon

## Global
- pokemon currently in play

## Summary
- each player only has 1 pokemon (no switching)
- the pokemon have fixed fast and charged attacks
- no shields to block charged attacks

### State
[hp, energy, turns until fast attack]

### Actions
[fast attack, charged attack 1, charged attack 2]

### Environment
- "get next state" function



# Sources

https://github.com/labmlai/annotated_deep_learning_paper_implementations/tree/master/labml_nn/rl/dqn
