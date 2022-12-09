import torch

from .pokemon import Pokemon


class Battle:
    def __init__(self, pokemon_1: Pokemon, pokemon_2: Pokemon) -> None:
        self.pokemon_1 = pokemon_1
        self.pokemon_2 = pokemon_2

    def __repr__(self) -> str:
        return f"""
Pokemon 1: {self.pokemon_1}
Pokemon 2: {self.pokemon_2}
        """

    def get_state(self):
        return self.pokemon_1.get_state(self.pokemon_2) / 200, self.pokemon_2.get_state(self.pokemon_1) / 200

    def update(self, action_1: int, action_2: int) -> None:
        if action_1 == 1:
            damage_1 = self.pokemon_1.fast_attack_damage(self.pokemon_2)
        elif action_1 == 2:
            damage_1 = self.pokemon_1.charged_attack_1_damage(self.pokemon_2)
        elif action_1 == 3:
            damage_1 = self.pokemon_1.charged_attack_2_damage(self.pokemon_2)
        else:
            damage_1 = 0

        if action_2 == 1:
            damage_2 = self.pokemon_2.fast_attack_damage(self.pokemon_1)
        elif action_2 == 2:
            damage_2 = self.pokemon_2.charged_attack_2_damage(self.pokemon_1)
        elif action_2 == 3:
            damage_2 = self.pokemon_2.charged_attack_2_damage(self.pokemon_1)
        else:
            damage_2 = 0

        self.pokemon_1.update(action_1, damage_2)
        self.pokemon_2.update(action_2, damage_1)

    def get_reward(self, action: int) -> float:
        if self.pokemon_2.hp <= 0:
            reward = 200.0
        elif action == 1:
            reward = self.pokemon_1.fast_attack_damage(self.pokemon_2)  # fast attack damage
        elif action == 2:
            reward = self.pokemon_1.charged_attack_1_damage(self.pokemon_2)  # charged attack 1 damage
        elif action == 3:
            reward = self.pokemon_1.charged_attack_2_damage(self.pokemon_2)  # charged attack 2 damage
        else:
            reward = 0
        return reward / 200

    def done(self) -> bool:
        return self.pokemon_1.hp <= 0 or self.pokemon_2.hp <= 0

    def win(self) -> bool:
        return self.pokemon_1.hp > 0 and self.pokemon_2.hp <= 0
