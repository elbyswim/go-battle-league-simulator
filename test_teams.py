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


mud_shot = FastAttack(*fast_moves_df[fast_moves_df["Move"] == "Mud Shot"].values[0])
shadow_claw = FastAttack(*fast_moves_df[fast_moves_df["Move"] == "Shadow Claw"].values[0])
charm = FastAttack(*fast_moves_df[fast_moves_df["Move"] == "Charm"].values[0])
smack_down = FastAttack(*fast_moves_df[fast_moves_df["Move"] == "Smack Down"].values[0])
counter = FastAttack(*fast_moves_df[fast_moves_df["Move"] == "Counter"].values[0])
powder_snow = FastAttack(*fast_moves_df[fast_moves_df["Move"] == "Powder Snow"].values[0])
wing_attack = FastAttack(*fast_moves_df[fast_moves_df["Move"] == "Wing Attack"].values[0])
lick = FastAttack(*fast_moves_df[fast_moves_df["Move"] == "Lick"].values[0])
poison_sting = FastAttack(*fast_moves_df[fast_moves_df["Move"] == "Poison Sting"].values[0])
vine_whip = FastAttack(*fast_moves_df[fast_moves_df["Move"] == "Vine Whip"].values[0])
dragon_breath = FastAttack(*fast_moves_df[fast_moves_df["Move"] == "Dragon Breath"].values[0])
spark = FastAttack(*fast_moves_df[fast_moves_df["Move"] == "Spark"].values[0])
lock_on = FastAttack(*fast_moves_df[fast_moves_df["Move"] == "Lock On"].values[0])
bubble = FastAttack(*fast_moves_df[fast_moves_df["Move"] == "Bubble"].values[0])



rock_slide = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Rock Slide"].values[0, :4])
earthquake = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Earthquake"].values[0, :4])
flamethrower = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Flamethrower"].values[0, :4])
stone_edge = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Stone Edge"].values[0, :4])
weather_ball_ice = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Weather Ball (Ice)"].values[0, :4])
psyshock = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Psyshock"].values[0, :4])
seed_bomb = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Seed Bomb"].values[0, :4])
shadow_ball = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Shadow Ball"].values[0, :4])
foul_play = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Foul Play"].values[0, :4])
return1 = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Return"].values[0, :4])
ice_punch = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Ice Punch"].values[0, :4])
psychic = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Psychic"].values[0, :4])
energy_ball = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Energy Ball"].values[0, :4])
sky_attack = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Sky Attack"].values[0, :4])
body_slam = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Body Slam"].values[0, :4])
power_whip = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Power Whip"].values[0, :4])
brine = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Brine"].values[0, :4])
sludge_wave = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Sludge Wave"].values[0, :4])
frenzy_plant = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Frenzy Plant"].values[0, :4])
sludge_bomb = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Sludge Bomb"].values[0, :4])
moonblast = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Moonblast"].values[0, :4])
hydro_cannon = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Hydro Cannon"].values[0, :4])
surf = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Surf"].values[0, :4])
thunderbolt = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Thunderbolt"].values[0, :4])
focus_blast = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Focus Blast"].values[0, :4])
zap_cannon = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Zap Cannon"].values[0, :4])
ice_beam = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Ice Beam"].values[0, :4])
play_rough = ChargedAttack(*charged_moves_df[charged_moves_df["Move"] == "Play Rough"].values[0, :4])

Trevenant = Pokemon(
    *pokemon_df[pokemon_df["Name"] == "Trevenant"].values[0, :6],
    dict(zip(pokemon_df.columns[6:], pokemon_df[pokemon_df["Name"] == "Trevenant"].values[0, 6:])),
    shadow_claw, seed_bomb, shadow_ball
)
Ninetales = Pokemon(
    *pokemon_df[pokemon_df["Name"] == "Ninetales (Alolan)"].values[0, :6],
    dict(zip(pokemon_df.columns[6:], pokemon_df[pokemon_df["Name"] == "Ninetales (Alolan)"].values[0, 6:])),
    charm, weather_ball_ice, psyshock
)
Stunfisk = Pokemon(
    *pokemon_df[pokemon_df["Name"] == "Stunfisk (Galarian)"].values[0, :6],
    dict(zip(pokemon_df.columns[6:], pokemon_df[pokemon_df["Name"] == "Stunfisk (Galarian)"].values[0, 6:])),
    mud_shot, rock_slide, earthquake
)
Bastiodon = Pokemon(
    *pokemon_df[pokemon_df["Name"] == "Bastiodon"].values[0, :6],
    dict(zip(pokemon_df.columns[6:], pokemon_df[pokemon_df["Name"] == "Bastiodon"].values[0, 6:])),
    smack_down, stone_edge, flamethrower
)
Medicham = Pokemon(
    *pokemon_df[pokemon_df["Name"] == "Medicham"].values[0, :6],
    dict(zip(pokemon_df.columns[6:], pokemon_df[pokemon_df["Name"] == "Medicham"].values[0, 6:])),
    counter, ice_punch, psychic
)
Sableye = Pokemon(
    *pokemon_df[pokemon_df["Name"] == "Sableye"].values[0, :6],
    dict(zip(pokemon_df.columns[6:], pokemon_df[pokemon_df["Name"] == "Sableye"].values[0, 6:])),
    shadow_claw, foul_play, return1
)
Abomasnow = Pokemon(
    *pokemon_df[pokemon_df["Name"] == "Abomasnow"].values[0, :6],
    dict(zip(pokemon_df.columns[6:], pokemon_df[pokemon_df["Name"] == "Abomasnow"].values[0, 6:])),
    powder_snow, weather_ball_ice, energy_ball
)
Noctowl = Pokemon(
    *pokemon_df[pokemon_df["Name"] == "Noctowl"].values[0, :6],
    dict(zip(pokemon_df.columns[6:], pokemon_df[pokemon_df["Name"] == "Noctowl"].values[0, 6:])),
    wing_attack, sky_attack, shadow_ball
)
Lickitung = Pokemon(
    *pokemon_df[pokemon_df["Name"] == "Lickitung"].values[0, :6],
    dict(zip(pokemon_df.columns[6:], pokemon_df[pokemon_df["Name"] == "Lickitung"].values[0, 6:])),
    lick, body_slam, power_whip
)
Toxapex = Pokemon(
    *pokemon_df[pokemon_df["Name"] == "Toxapex"].values[0, :6],
    dict(zip(pokemon_df.columns[6:], pokemon_df[pokemon_df["Name"] == "Toxapex"].values[0, 6:])),
    poison_sting, brine, sludge_wave
)
Meganium = Pokemon(
    *pokemon_df[pokemon_df["Name"] == "Meganium"].values[0, :6],
    dict(zip(pokemon_df.columns[6:], pokemon_df[pokemon_df["Name"] == "Meganium"].values[0, 6:])),
    vine_whip, frenzy_plant, earthquake
)
Venusaur = Pokemon(
    *pokemon_df[pokemon_df["Name"] == "Venusaur"].values[0, :6],
    dict(zip(pokemon_df.columns[6:], pokemon_df[pokemon_df["Name"] == "Venusaur"].values[0, 6:])),
    vine_whip, frenzy_plant, sludge_bomb
)
Altaria = Pokemon(
    *pokemon_df[pokemon_df["Name"] == "Altaria"].values[0, :6],
    dict(zip(pokemon_df.columns[6:], pokemon_df[pokemon_df["Name"] == "Altaria"].values[0, 6:])),
    dragon_breath, sky_attack, moonblast
)
Swampert = Pokemon(
    *pokemon_df[pokemon_df["Name"] == "Swampert"].values[0, :6],
    dict(zip(pokemon_df.columns[6:], pokemon_df[pokemon_df["Name"] == "Swampert"].values[0, 6:])),
    mud_shot, hydro_cannon, earthquake
)
Lanturn = Pokemon(
    *pokemon_df[pokemon_df["Name"] == "Lanturn"].values[0, :6],
    dict(zip(pokemon_df.columns[6:], pokemon_df[pokemon_df["Name"] == "Lanturn"].values[0, 6:])),
    spark, surf, thunderbolt
)
Registeel = Pokemon(
    *pokemon_df[pokemon_df["Name"] == "Registeel"].values[0, :6],
    dict(zip(pokemon_df.columns[6:], pokemon_df[pokemon_df["Name"] == "Registeel"].values[0, 6:])),
    lock_on, focus_blast, zap_cannon
)
Azumarill = Pokemon(
    *pokemon_df[pokemon_df["Name"] == "Azumarill"].values[0, :6],
    dict(zip(pokemon_df.columns[6:], pokemon_df[pokemon_df["Name"] == "Azumarill"].values[0, 6:])),
    bubble, ice_beam, play_rough
)

team1 = Team(Trevenant, Ninetales, Stunfisk)
team2 = Team(Bastiodon, Medicham, Sableye)
team3 = Team(Abomasnow, Medicham, Sableye)
team4 = Team(Trevenant, Noctowl, Stunfisk)
team5 = Team(Medicham, Lickitung, Toxapex)
team6 = Team(Bastiodon, Meganium, Trevenant)
team7 = Team(Venusaur, Sableye, Stunfisk)
team8 = Team(Trevenant, Altaria, Stunfisk)
team9 = Team(Altaria, Lickitung, Swampert)
team10 = Team(Altaria, Lanturn, Registeel)
team11 = Team(Stunfisk, Azumarill, Trevenant)
team12 = Team(Trevenant, Azumarill, Stunfisk)
team13 = Team(Ninetales, Sableye, Swampert)
team14 = Team(Altaria, Azumarill, Stunfisk)
team15 = Team(Stunfisk, Altaria, Trevenant)
team16 = Team(Stunfisk, Sableye, Venusaur)
team17 = Team(Trevenant, Stunfisk, Toxapex)
team18 = Team(Medicham, Lickitung, Ninetales)
team19 = Team(Azumarill, Medicham, Registeel)
team20 = Team(Medicham, Bastiodon, Lickitung)