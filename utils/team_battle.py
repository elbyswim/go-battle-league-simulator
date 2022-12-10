from .team import Team

class TeamBattle:
    def __init__(self, team_1: Team, team_2: Team) -> None:
        self.team_1 = team_1
        self.team_2 = team_2

    def __repr__(self) -> str:
        return f"""
Team 1: {self.team_1}
Team 2: {self.team_2}
        """

    def get_state(self):
        return self.team_1.get_state(self.team_2) / 200, self.team_2.get_state(self.team_1) / 200

    def update(self, action_1: int, action_2: int) -> None:
        if action_1 == 1:
            damage_1 = self.team_1.pokemon_1.fast_attack_damage(self.team_2.pokemon_1)
        elif action_1 == 2:
            damage_1 = self.team_1.pokemon_1.charged_attack_1_damage(self.team_2.pokemon_1)
        elif action_1 == 3:
            damage_1 = self.team_1.pokemon_1.charged_attack_2_damage(self.team_2.pokemon_1)
        else:
            damage_1 = 0

        if action_2 == 1:
            damage_2 = self.team_2.pokemon_2.fast_attack_damage(self.team_1.pokemon_1)
        elif action_2 == 2:
            damage_2 = self.team_2.pokemon_2.charged_attack_2_damage(self.team_1.pokemon_1)
        elif action_2 == 3:
            damage_2 = self.team_2.pokemon_2.charged_attack_2_damage(self.team_1.pokemon_1)
        else:
            damage_2 = 0

        self.team_1.update(action_1, action_2, damage_2)
        self.team_2.update(action_2, action_1, damage_1)

    def get_reward(self, action: int) -> float:
        possible_actions = self.team_1.get_actions()
        if self.team_2.lost():
            reward = 200
        elif action == 1:
            if any(possible_actions[0, 2:4]):
                reward = -10
            else:
                reward = self.team_1.pokemon_1.fast_attack_damage(self.team_2.pokemon_1)  # fast attack damage
                # reward = 1
        elif action == 2:
            reward = self.team_1.pokemon_1.charged_attack_1_damage(self.team_2.pokemon_1)  # charged attack 1 damage
        elif action == 3:
            reward = self.team_1.pokemon_1.charged_attack_2_damage(self.team_2.pokemon_1)  # charged attack 2 damage
        else:
            if any(possible_actions[0, 1:4]):
                reward = -10
            else:
                reward = -1
        return reward / 200

    def done(self) -> bool:
        return self.team_1.lost() or self.team_2.lost()

    def win(self) -> bool:
        return self.team_2.lost()
