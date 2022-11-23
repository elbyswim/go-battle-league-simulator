from q_learning import *
from utils import *
from torch import concatenate, zeros, rand, argmax, tensor

dqn = DQN()
mud_shot = FastAttack(3, 9, 2)
rock_slide = ChargedAttack(75, 45)
earthquake = ChargedAttack(120, 65)
lock_on = FastAttack(1, 5, 1)
focus_blast = ChargedAttack(150, 75)
flash_cannon = ChargedAttack(110, 70)

Stunfisk = Pokemon(177, 99.9, 127, mud_shot, rock_slide, earthquake)
Registeel = Pokemon(132, 93.8, 192.9, lock_on, focus_blast, flash_cannon)

agent = QLearning()

agent.learn([Stunfisk, Registeel])
