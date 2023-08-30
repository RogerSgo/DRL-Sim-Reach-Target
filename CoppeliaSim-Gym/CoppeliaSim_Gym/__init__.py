# Registro del entorno

from gymnasium.envs.registration import register


register(
    id="CoppeliaSim_Gym/GymCoppManR-v0",
    entry_point="CoppeliaSim_Gym.envs:GymCoppManR",
)