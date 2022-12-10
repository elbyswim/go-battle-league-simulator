from __future__ import annotations
from csv import reader
from itertools import count
import pandas as pd

from utils.attacks import FastAttack, ChargedAttack
from utils.battle import Battle
from utils.pokemon import *
from utils.team import *

# read data
pokemon_df = pd.read_csv("pokemon.csv")
fast_moves_df = pd.read_csv("fastmoves.csv")
charged_moves_df = pd.read_csv("chargedmoves.csv")
charged_moves_df = charged_moves_df[charged_moves_df.Effects.isnull()]

attack_dict = {}

attack_dict["mud_shot"] = FastAttack(*fast_moves_df[fast_moves_df["Move"] == "Mud Shot"].values[0])
attack_dict["shadow_claw"] = FastAttack(*fast_moves_df[fast_moves_df["Move"] == "Shadow Claw"].values[0])
attack_dict["charm"] = FastAttack(*fast_moves_df[fast_moves_df["Move"] == "Charm"].values[0])
attack_dict["smack_down"] = FastAttack(*fast_moves_df[fast_moves_df["Move"] == "Smack Down"].values[0])
attack_dict["counter"] = FastAttack(*fast_moves_df[fast_moves_df["Move"] == "Counter"].values[0])
attack_dict["powder_snow"] = FastAttack(*fast_moves_df[fast_moves_df["Move"] == "Powder Snow"].values[0])
attack_dict["wing_attack"] = FastAttack(*fast_moves_df[fast_moves_df["Move"] == "Wing Attack"].values[0])
attack_dict["lick"] = FastAttack(*fast_moves_df[fast_moves_df["Move"] == "Lick"].values[0])
attack_dict["poison_sting"] = FastAttack(*fast_moves_df[fast_moves_df["Move"] == "Poison Sting"].values[0])
attack_dict["vine_whip"] = FastAttack(*fast_moves_df[fast_moves_df["Move"] == "Vine Whip"].values[0])
attack_dict["dragon_breath"] = FastAttack(*fast_moves_df[fast_moves_df["Move"] == "Dragon Breath"].values[0])
attack_dict["spark"] = FastAttack(*fast_moves_df[fast_moves_df["Move"] == "Spark"].values[0])
attack_dict["lock_on"] = FastAttack(*fast_moves_df[fast_moves_df["Move"] == "Lock On"].values[0])
attack_dict["bubble"] = FastAttack(*fast_moves_df[fast_moves_df["Move"] == "Bubble"].values[0])

# charged attacks
attack_dict["rock_slide"] = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Rock Slide"].values[0, :4])
attack_dict["earthquake"] = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Earthquake"].values[0, :4])
attack_dict["flamethrower"] = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Flamethrower"].values[0, :4])
attack_dict["stone_edge"] = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Stone Edge"].values[0, :4])
attack_dict["weather_ball_ice"] = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Weather Ball (Ice)"].values[0, :4])
attack_dict["psyshock"] = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Psyshock"].values[0, :4])
attack_dict["seed_bomb"] = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Seed Bomb"].values[0, :4])
attack_dict["shadow_ball"] = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Shadow Ball"].values[0, :4])
attack_dict["foul_play"] = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Foul Play"].values[0, :4])
attack_dict["return1"] = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Return"].values[0, :4])
attack_dict["ice_punch"] = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Ice Punch"].values[0, :4])
attack_dict["psychic"] = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Psychic"].values[0, :4])
attack_dict["energy_ball"] = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Energy Ball"].values[0, :4])
attack_dict["sky_attack"] = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Sky Attack"].values[0, :4])
attack_dict["body_slam"] = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Body Slam"].values[0, :4])
attack_dict["power_whip"] = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Power Whip"].values[0, :4])
attack_dict["brine"] = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Brine"].values[0, :4])
attack_dict["sludge_wave"] = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Sludge Wave"].values[0, :4])
attack_dict["frenzy_plant"] = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Frenzy Plant"].values[0, :4])
attack_dict["sludge_bomb"] = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Sludge Bomb"].values[0, :4])
attack_dict["moonblast"] = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Moonblast"].values[0, :4])
attack_dict["hydro_cannon"] = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Hydro Cannon"].values[0, :4])
attack_dict["surf"] = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Surf"].values[0, :4])
attack_dict["thunderbolt"] = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Thunderbolt"].values[0, :4])
attack_dict["focus_blast"] = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Focus Blast"].values[0, :4])
attack_dict["zap_cannon"] = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Zap Cannon"].values[0, :4])
attack_dict["ice_beam"] = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Ice Beam"].values[0, :4])
attack_dict["play_rough"] = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Play Rough"].values[0, :4])

pokemon_dict = {}

pokemon_dict["Trevenant"] = ("shadow_claw", "seed_bomb", "shadow_ball")
pokemon_dict["Ninetales"] = ("charm", "weather_ball_ice", "psyshock")
pokemon_dict["Stunfisk"] = ("mud_shot", "rock_slide", "earthquake")
pokemon_dict["Bastiodon"] = ("smack_down", "stone_edge", "flamethrower")
pokemon_dict["Medicham"] = ("counter", "ice_punch", "psychic")

pokemon_dict["Sableye"] = ("shadow_claw", "foul_play", "return1")
pokemon_dict["Abomasnow"] = ("powder_snow", "weather_ball_ice", "energy_ball")
pokemon_dict["Noctowl"] = ("wing_attack", "sky_attack", "shadow_ball")
pokemon_dict["Lickitung"] = ("lick", "body_slam", "power_whip")
pokemon_dict["Toxapex"] = ("poison_sting", "brine", "sludge_wave")

pokemon_dict["Meganium"] = ("vine_whip", "frenzy_plant", "earthquake")
pokemon_dict["Venusaur"] = ("vine_whip", "frenzy_plant", "sludge_bomb")
pokemon_dict["Altaria"] = ("dragon_breath", "sky_attack", "moonblast")
pokemon_dict["Swampert"] = ("mud_shot", "hydro_cannon", "earthquake")
pokemon_dict["Lanturn"] = ("spark", "surf", "thunderbolt")

pokemon_dict["Registeel"] = ("lock_on", "focus_blast", "zap_cannon")
pokemon_dict["Azumarill"] = ("bubble", "ice_beam", "play_rough")

def initiate_pokemon(name: str) -> Pokemon:
    poke_info = pokemon_dict[name]

    return Pokemon(
        *pokemon_df[pokemon_df["Name"] == name].values[0, :6],
        dict(zip(pokemon_df.columns[6:], pokemon_df[pokemon_df["Name"] == name].values[0, 6:])),
        *[attack_dict[attack] for attack in poke_info]
    )

def make_team(team_class):
    pokemon = []
    for i in range(3):
        pokemon_name = random.choice(list(pokemon_dict.keys()))
        pokemon.append(initiate_pokemon(pokemon_name))
    return team_class(*pokemon)

# team1 = Team(Trevenant, Ninetales, Stunfisk)
# team2 = Team(Bastiodon, Medicham, Sableye)
# team2 = TeamFastAttack(Bastiodon, Medicham, Sableye)
# team3 = Team(Trevenant1, Ninetales1, Stunfisk1)
# team4 = Team(Bastiodon1, Medicham1, Sableye1)
# team3 = Team(Abomasnow, Medicham, Sableye)
# team4 = Team(Trevenant, Noctowl, Stunfisk)
# team5 = Team(Medicham, Lickitung, Toxapex)
# team6 = Team(Bastiodon, Meganium, Trevenant)
# team7 = Team(Venusaur, Sableye, Stunfisk)
# team8 = Team(Trevenant, Altaria, Stunfisk)
# team9 = Team(Altaria, Lickitung, Swampert)
# team10 = Team(Altaria, Lanturn, Registeel)
# team11 = Team(Stunfisk, Azumarill, Trevenant)
# team12 = Team(Trevenant, Azumarill, Stunfisk)
# team13 = Team(Ninetales, Sableye, Swampert)
# team14 = Team(Altaria, Azumarill, Stunfisk)
# team15 = Team(Stunfisk, Altaria, Trevenant)
# team16 = Team(Stunfisk, Sableye, Venusaur)
# team17 = Team(Trevenant, Stunfisk, Toxapex)
# team18 = Team(Medicham, Lickitung, Ninetales)
# team19 = Team(Azumarill, Medicham, Registeel)
# team20 = Team(Medicham, Bastiodon, Lickitung)