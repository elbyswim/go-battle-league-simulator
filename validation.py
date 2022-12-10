from q_learning import *
from simulations import *
from test_teams import *
from utils.pokemon import *
from utils.team import *

def main():
    agent = QLearningTeam(input_dim=72, n_actions=6, hidden_size=32, batch_size=32, lr=1e-5, weight_decay=1e-7)
    agent.dqn.load_state_dict(torch.load("model.pt"))

    simulate_team_battle(agent, make_team(Team), make_team(TeamNoAttack))
    simulate_team_battle(agent, make_team(Team), make_team(TeamFastAttack))
    simulate_team_battle(agent, make_team(Team), make_team(Team))

if __name__ == "__main__":
    main()
