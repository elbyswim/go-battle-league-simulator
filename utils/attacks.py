class FastAttack:
    def __init__(
        self,
        name: str,
        damage: int,
        energy_generated: int,
        cooldown: int,
        type: str
    ) -> None:
        self.name = name
        self.damage = damage
        self.energy_generated = energy_generated
        self.cooldown = cooldown - 1
        self.type = type.capitalize()

    def __repr__(self) -> str:
        return f"Name: {self.name}, type: {self.type}, damage: {self.damage}, energy generated: {self.energy_generated}, cooldown: {self.cooldown + 1}"


class ChargedAttack:
    def __init__(
        self,
        name: str,
        damage: int,
        energy_cost: int,
        type: str
    ) -> None:
        self.name = name
        self.damage = damage
        self.energy_cost = energy_cost
        self.type = type.capitalize()

    def __repr__(self) -> str:
        return f"Name: {self.name}, type: {self.type}, damage: {self.damage}, energy cost: {self.energy_cost}"
