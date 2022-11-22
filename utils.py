import torch
import math


class FastAttack:
    def __init__(self, damage: int, energy_generated: int, turns: int) -> None:
        self.damage = damage
        self.energy_generated = energy_generated
        self.turns = turns


class ChargedAttack:
    def __init__(self, damage: int, energy_cost: int) -> None:
        self.damage = damage
        self.energy_cost = energy_cost


class Pokemon:
    def __init__(
        self,
        initial_hp: int,
        attack: float,
        defense: float,
        fast_attack: FastAttack,
        charged_attack_1: ChargedAttack,
        charged_attack_2: ChargedAttack,
        # type_effectiveness: dict
    ) -> None:
        self.initial_hp = initial_hp
        self.attack = attack
        self.defense = defense
        self.fast_attack = fast_attack
        self.charged_attack_1 = charged_attack_1
        self.charged_attack_2 = charged_attack_2
        # self.type_effectiveness = type_effectiveness

    def get_initial_state(self):
        return torch.tensor([self.initial_hp, 0, 0])

    def get_actions(self, state):
        return torch.tensor(
            [
                # fast attack
                1 if state[2] == 0 else 0,
                # charged attack 1
                1 if state[1] >= self.charged_attack_1.energy_cost else 0,
                # charged attack 2
                1 if state[1] >= self.charged_attack_2.energy_cost else 0,
            ]
        )

    def fast_attack_damage(self, defender: Pokemon):
        # effectiveness = defender.type_effectiveness[self.fast_attack]
        bonus_multiplier = 0.65
        damage = math.floor(self.fast_attack.damage * self.attack / defender.defense * bonus_multiplier)
        # need to add stab and effectiveness
        return damage + 1

    def charged_attack_1_damage(self, defender: Pokemon, charge: float = 1):
        bonus_multiplier = 0.65
        damage = math.floor(self.charged_attack_1.damage * self.attack / defender.defense * bonus_multiplier * charge)
        return damage + 1

    def charged_attack_2_damage(self, defender: Pokemon, charge: float = 1):
        bonus_multiplier = 0.65
        damage = math.floor(self.charged_attack_2.damage * self.attack / defender.defense * bonus_multiplier * charge)
        return damage + 1


class Battle:
    def __init__(self, pokemon_1: Pokemon, pokemon_2: Pokemon) -> None:
        self.pokemon_1 = pokemon_1
        self.pokemon_2 = pokemon_2

    def get_initial_state(self):
        return torch.tensor([self.pokemon_1.initial_hp, 0, 0, self.pokemon_2.initial_hp, 0, 0])

    def get_next_state(
        self,
        current_state: torch.Tensor,
        action_1: torch.Tensor,
        action_2: torch.Tensor,
    ):
        # state:
        # 0: pokemon 1 hp
        # 1: pokemon 1 energy
        # 2: pokemon 1 cooldown
        # 3: pokemon 2 hp
        # 4: pokemon 2 energy
        # 5: pokemon 2 cooldown

        next_state = current_state
        # changes based on pokemon 1 actions
        # fast attack
        if action_1[0] == 1:
            next_state[1] += self.fast_attack.energy_generated
            next_state[2] += self.fast_attack.turns
            next_state[3] -= self.pokemon_1.fast_attack_damage(self.pokemon_2)
        else:
            next_state[2] -= 1
        # Charged move 1
        if action_1[1] == 1:
            next_state[1] -= self.charged_attack_1.energy_cost
            next_state[3] -= self.pokemon_1.charged_attack_1_damage(self.pokemon_2)
        # Charged move 2
        if action_1[2] == 1:
            next_state[1] -= self.charged_attack_2.energy_cost
            next_state[3] -= self.pokemon_1.charged_attack_2_damage(self.pokemon_2)
        # changes based on pokemon 2 actions
        # fast attack
        if action_2[0] == 1:
            next_state[4] += self.fast_attack.energy_generated
            next_state[5] += self.fast_attack.turns
            next_state[0] -= self.pokemon_2.fast_attack_damage(self.pokemon_1)
        else:
            next_state[5] -= 1
        # Charged move 1
        if action_2[1] == 1:
            next_state[4] -= self.charged_attack_1.energy_cost
            next_state[0] -= self.pokemon_2.charged_attack_1_damage(self.pokemon_1)
        # Charged move 2
        if action_2[2] == 1:
            next_state[4] -= self.charged_attack_2.energy_cost
            next_state[0] -= self.pokemon_2.charged_attack_2_damage(self.pokemon_1)
        return next_state

    def get_reward(self, state: torch.Tensor):
        return 1 if state[3] <= 0 else 0