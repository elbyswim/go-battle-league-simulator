from __future__ import annotations

import torch

from .pokemon import Pokemon


COOLDOWN = 30


class Team:
    def __init__(self, pokemon_1: Pokemon, pokemon_2: Pokemon, pokemon_3: Pokemon):
        self.pokemon_1 = pokemon_1
        self.pokemon_2 = pokemon_2
        self.pokemon_3 = pokemon_3
        self.switch_cooldown = 0
    
    def __repr__(self) -> str:
        return f"""
Switch Cooldown: {self.switch_cooldown}
Pokemon 1: {self.pokemon_1}
Pokemon 2: {self.pokemon_2}
Pokemon 3: {self.pokemon_3}
        """

    def get_state(self, defending_team: Team) -> torch.Tensor:
        return (
            torch.tensor(
                [
                    self.pokemon_1.hp,
                    self.pokemon_1.energy,
                    self.pokemon_1.cooldown,
                    self.pokemon_1.fast_attack_damage(defending_team.pokemon_1),
                    self.pokemon_1.charged_attack_1_damage(defending_team.pokemon_1),
                    self.pokemon_1.charged_attack_2_damage(defending_team.pokemon_1),
                    self.pokemon_1.fast_attack_damage(defending_team.pokemon_2),
                    self.pokemon_1.charged_attack_1_damage(defending_team.pokemon_2),
                    self.pokemon_1.charged_attack_2_damage(defending_team.pokemon_2),
                    self.pokemon_1.fast_attack_damage(defending_team.pokemon_3),
                    self.pokemon_1.charged_attack_1_damage(defending_team.pokemon_3),
                    self.pokemon_1.charged_attack_2_damage(defending_team.pokemon_3),

                    self.pokemon_2.hp,
                    self.pokemon_2.energy,
                    self.pokemon_2.cooldown,
                    self.pokemon_2.fast_attack_damage(defending_team.pokemon_1),
                    self.pokemon_2.charged_attack_1_damage(defending_team.pokemon_1),
                    self.pokemon_2.charged_attack_2_damage(defending_team.pokemon_1),
                    self.pokemon_2.fast_attack_damage(defending_team.pokemon_2),
                    self.pokemon_2.charged_attack_1_damage(defending_team.pokemon_2),
                    self.pokemon_2.charged_attack_2_damage(defending_team.pokemon_2),
                    self.pokemon_2.fast_attack_damage(defending_team.pokemon_3),
                    self.pokemon_2.charged_attack_1_damage(defending_team.pokemon_3),
                    self.pokemon_2.charged_attack_2_damage(defending_team.pokemon_3),

                    self.pokemon_3.hp,
                    self.pokemon_3.energy,
                    self.pokemon_3.cooldown,
                    self.pokemon_3.fast_attack_damage(defending_team.pokemon_1),
                    self.pokemon_3.charged_attack_1_damage(defending_team.pokemon_1),
                    self.pokemon_3.charged_attack_2_damage(defending_team.pokemon_1),
                    self.pokemon_3.fast_attack_damage(defending_team.pokemon_2),
                    self.pokemon_3.charged_attack_1_damage(defending_team.pokemon_2),
                    self.pokemon_3.charged_attack_2_damage(defending_team.pokemon_2),
                    self.pokemon_3.fast_attack_damage(defending_team.pokemon_3),
                    self.pokemon_3.charged_attack_1_damage(defending_team.pokemon_3),
                    self.pokemon_3.charged_attack_2_damage(defending_team.pokemon_3),
                ]
            )
            .float()
            .reshape(1, -1)
        )

    def update(self, action: int, opponent_action: int, damage_taken: float) -> None:
        self.pokemon_1.hp -= damage_taken
        if action == 0:
            self.pokemon_1.cooldown -= 1
        # fast attack
        if action == 1:
            self.pokemon_1.energy += self.pokemon_1.fast_attack.energy_generated
            self.pokemon_1.cooldown = self.pokemon_1.fast_attack.cooldown
        # Charged move 1
        if action == 2:
            self.pokemon_1.energy -= self.pokemon_1.charged_attack_1.energy_cost
        # Charged move 2
        if action == 3:
            self.pokemon_1.energy -= self.pokemon_1.charged_attack_2.energy_cost
        # Switch 2
        if action == 4:
            self.pokemon_1, self.pokemon_2 = self.pokemon_2, self.pokemon_1
            self.switch_cooldown = COOLDOWN
        # Switch 3
        if action == 5:
            self.pokemon_1, self.pokemon_3 = self.pokemon_3, self.pokemon_1
            self.switch_cooldown = COOLDOWN
        if opponent_action in {2, 3}:
            self.pokemon_1.cooldown = 0

    def get_actions(self) -> torch.Tensor:
        # actions: nothing, fast, charge1, charge2, switch2, switch3
        # fast attack
        if self.pokemon_1.hp > 0:
            actions = torch.ones(6)
            if self.pokemon_1.cooldown > 0:
                actions[1] = 0
            # charged attack 1
            if self.pokemon_1.energy < self.pokemon_1.charged_attack_1.energy_cost:
                actions[2] = 0
            # charged attack 2
            if self.pokemon_1.energy < self.pokemon_1.charged_attack_2.energy_cost:
                actions[3] = 0
            # switch
            if self.switch_cooldown > 0 or self.pokemon_1.cooldown > 0 or self.pokemon_2.hp < 0: 
                actions[4] = 0
            if self.switch_cooldown > 0 or self.pokemon_1.cooldown > 0 or self.pokemon_3.hp < 0: 
                actions[5] = 0
        else:
            actions = torch.zeros(6)
            if self.pokemon_2.hp > 0:
                actions[4] = 1
            if self.pokemon_3.hp > 0:
                actions[5] = 1
        return actions.reshape(1, 6).bool()

    def lost(self):
        return self.pokemon_1.hp <= 0 and self.pokemon_2.hp <= 0 and self.pokemon_3.hp <= 0


class HiddenTeam(Team):
    def get_state(self, defending_team: Team) -> torch.Tensor:
        return (
            torch.tensor(
                [
                    self.pokemon_1.hp,
                    self.pokemon_1.energy,
                    self.pokemon_1.cooldown,
                    self.pokemon_1.fast_attack_damage(defending_team.pokemon_1),
                    self.pokemon_1.charged_attack_1_damage(defending_team.pokemon_1),
                    self.pokemon_1.charged_attack_2_damage(defending_team.pokemon_1),

                    self.pokemon_2.hp,
                    self.pokemon_2.energy,
                    self.pokemon_2.cooldown,
                    self.pokemon_2.fast_attack_damage(defending_team.pokemon_1),
                    self.pokemon_2.charged_attack_1_damage(defending_team.pokemon_1),
                    self.pokemon_2.charged_attack_2_damage(defending_team.pokemon_1),

                    self.pokemon_3.hp,
                    self.pokemon_3.energy,
                    self.pokemon_3.cooldown,
                    self.pokemon_3.fast_attack_damage(defending_team.pokemon_1),
                    self.pokemon_3.charged_attack_1_damage(defending_team.pokemon_1),
                    self.pokemon_3.charged_attack_2_damage(defending_team.pokemon_1),
                ]
            )
            .float()
            .reshape(1, -1)
        )