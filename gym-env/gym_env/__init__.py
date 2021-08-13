from gym.envs.registration import register

register(
    id="env-v0", entry_point="gym_env.envs:FooEnv",
)
register(
    id="env-extrahard-v0", entry_point="gym_env.envs:FooExtraHardEnv",
)
