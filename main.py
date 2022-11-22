import q_learning
import utils

dqn = q_learning.DQN()
f1 = utils.FastAttack(0, 0, 0)
c1 = utils.ChargedAttack(0, 0)
c2 = utils.ChargedAttack(0, 0)
p1 = utils.Pokemon(100, 0, 0, f1, c1, c2)
s_0 = p1.get_initial_state()

print(s_0)

for i in range(4):
    print(q_learning.action_to_tensor(i))

print(concatenate(s_0, ))

for i in range(10):
    print(q_learning.select_action(s_0, p1, dqn))