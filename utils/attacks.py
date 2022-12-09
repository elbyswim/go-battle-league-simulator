class FastAttack:
    def __init__(
        self,
        name: str,
        damage: int,
        energy_generated: int,
        cooldown: int
        # , type: str
    ) -> None:
        self.name = name
        self.damage = damage
        self.energy_generated = energy_generated
        self.cooldown = cooldown
        # self.type = type

    def __repr__(self) -> str:
        return f"Name: {self.name}, damage: {self.damage}, energy generated: {self.energy_generated}, cooldown: {self.cooldown}"


class ChargedAttack:
    def __init__(
        self,
        name: str,
        damage: int,
        energy_cost: int
        # , type: str
    ) -> None:
        self.name = name
        self.damage = damage
        self.energy_cost = energy_cost
        # self.type = type

    def __repr__(self) -> str:
        return f"Name: {self.name}, damage: {self.damage}, energy cost: {self.energy_cost}"
