"""Generate one paired root, replay C1, and save the final frame."""

from pathlib import Path

from switchdoor import C1, GenerationConfig, SwitchDoorEnv, build_scenario_group

group = build_scenario_group(GenerationConfig(seed=7), candidate_index=0)
env = SwitchDoorEnv(group.shell.spec, C1)
env.rollout(group.shell.actions)
Path("switchdoor_example.png").write_bytes(env.render_png())
print(C1.public_dict(), env.state.to_dict())
