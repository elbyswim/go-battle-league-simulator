from __future__ import annotations
import math
import random

import torch

from .attacks import FastAttack, ChargedAttack


class Pokemon:
    def __init__(
        self,
        name: str,
        initial_hp: float,
        attack: float,
        defense: float,
        # type_1 = str
        # type_2 = str
        # type_effectiveness: dict
        fast_attack: FastAttack,
        charged_attack_1: ChargedAttack,
        charged_attack_2: ChargedAttack,
    ) -> None:
        self.name = name
        self.hp = initial_hp
        self.energy = 0
        self.cooldown = 0
        self.attack = attack
        self.defense = defense
        # self.type_1 = type_1
        # self.type_2 = type_2
        # self.type_effectiveness = type_effectiveness
        self.fast_attack = fast_attack
        self.charged_attack_1 = charged_attack_1
        self.charged_attack_2 = charged_attack_2

    def __repr__(self) -> str:
        return f"""
Name: {self.name}
HP: {self.hp}, Energy: {self.energy}, Cooldown: {self.cooldown}
Fast Attack: {self.fast_attack}
Charged Attack 1: {self.charged_attack_1}
Charged Attack 2: {self.charged_attack_2}
        """

    def get_state(self, defender: Pokemon) -> torch.Tensor:
        return (
            torch.tensor(
                [
                    self.hp,
                    self.energy,
                    self.cooldown,
                    self.fast_attack_damage(defender),
                    self.charged_attack_1_damage(defender),
                    self.charged_attack_2_damage(defender),
                ]
            )
            .float()
            .reshape(1, -1)
        )

    def update(self, action: int, damage_taken: float) -> None:
        self.hp -= damage_taken
        if action == 0:
            self.cooldown -= 1
        # fast attack
        if action == 1:
            self.energy += self.fast_attack.energy_generated
            self.cooldown = self.fast_attack.cooldown
        # Charged move 1
        if action == 2:
            self.energy -= self.charged_attack_1.energy_cost
        # Charged move 2
        if action == 3:
            self.energy -= self.charged_attack_2.energy_cost

    def get_actions(self) -> torch.Tensor:
        actions = torch.ones(4)
        # fast attack
        if self.cooldown > 0:
            actions[1] = 0
        # charged attack 1
        if self.energy < self.charged_attack_1.energy_cost:
            actions[2] = 0
        # charged attack 2
        if self.energy < self.charged_attack_2.energy_cost:
            actions[3] = 0
        return actions.reshape(1, 4).bool()

    def fast_attack_damage(self, defender: Pokemon) -> float:
        # effectiveness = defender.type_effectiveness[self.fast_attack.type]
        # if self.fast_attack.type == self.type_1 or self.fast_attack.type == self.type_2:
        #     stab = 1.2
        # else: stab = 1
        bonus_multiplier = 0.65
        damage = math.floor(self.fast_attack.damage * self.attack / defender.defense * bonus_multiplier)
        # need to also multiply by stab and effectiveness
        return damage + 1

    def charged_attack_1_damage(self, defender: Pokemon, charge: float = 1) -> float:
        # effectiveness = defender.type_effectiveness[self.charged_attack_1.type]
        # if self.charged_attack_1.type == self.type_1 or self.charged_attack_1.type == self.type_2:
        #     stab = 1.2
        # else: stab = 1
        bonus_multiplier = 0.65
        damage = math.floor(self.charged_attack_1.damage * self.attack / defender.defense * bonus_multiplier * charge)
        # need to also multiply by stab and effectiveness
        return damage + 1

    def charged_attack_2_damage(self, defender: Pokemon, charge: float = 1) -> float:
        # effectiveness = defender.type_effectiveness[self.charged_attack_1.type]
        # if self.charged_attack_2.type == self.type_1 or self.charged_attack_2.type == self.type_2:
        #     stab = 1.2
        # else: stab = 1
        bonus_multiplier = 0.65
        damage = math.floor(self.charged_attack_2.damage * self.attack / defender.defense * bonus_multiplier * charge)
        # need to also multiply by stab and effectiveness
        return damage + 1


class PokemonNoAttack(Pokemon):
    """Pokemon that never attacks."""

    def get_actions(self) -> torch.Tensor:
        return torch.tensor([1, 0, 0, 0]).reshape(1, 4).bool()


class PokemonFastAttack(Pokemon):
    """Pokemon that fast attacks when possible. Never uses charged attack."""

    def get_actions(self) -> torch.Tensor:
        return torch.tensor([self.cooldown > 0, self.cooldown <= 0, 0, 0]).reshape(1, 4).bool()


class PokemonChargedAttack(Pokemon):
    """
    Pokemon that uses charged attack when possible.
    Assumes the cheaper charged move is always the first.
    """

    def get_actions(self) -> torch.Tensor:
        # do charged move if able
        if self.energy >= self.charged_attack_1.energy_cost:
            action = 2
        # else do fast attack if able
        elif self.cooldown <= 0:
            action = 1
        else:
            action = 0
        actions = torch.zeros(4)
        actions[action] = 1
        return actions.reshape(1, 4).bool()


class PokemonRandomAction(Pokemon):
    """
    Pokemon that uses charged attack when possible.
    Assumes the cheaper charged move is always the first.
    """

    def get_actions(self) -> torch.Tensor:
        # do charged move if able
        actions = [0]
        # fast attack
        if self.cooldown <= 0:
            actions.append(1)
        # charged attack 1
        if self.energy >= self.charged_attack_1.energy_cost:
            actions.append(2)
        # charged attack 2
        if self.energy >= self.charged_attack_2.energy_cost:
            actions.append(3)
        actions_tensor = torch.zeros(4)
        actions_tensor[random.choice(actions)] = 1
        return actions_tensor.reshape(1, 4).bool()
