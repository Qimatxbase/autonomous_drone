# Autonomous Drone Navigation

PPO drone navigation with depth observations, staged curricula and precise arrival evaluation in PyBullet.

**Technologies:** Python · PyBullet · Gymnasium · Stable-Baselines3

## What is included

- Easy → medium → hard curricula with independent evaluation gates.
- Depth/state feature extraction, navigation rewards and obstacle avoidance.
- A separate 5 cm goal experiment and real Tello deployment helpers.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run

```bash
python -m gym_pybullet_drones.examples.curriculum_navigate --help
```

Saved checkpoints, full training runs and group report drafts remain local. Hardware deployment requires a configured Tello; the simulation can be explored independently.

## Checks

```bash
python -m pytest tests/test_build.py tests/test_curriculum.py tests/test_center_curriculum.py -q
```

## Source and credits

This portfolio snapshot preserves work from the following repositories. Existing author credits, citations and license files are retained; team and upstream code are not presented as solely authored work.

- https://github.com/Quang0309/AutoPilotDrone
- https://github.com/learnsyslab/gym-pybullet-drones

[Portfolio index](https://github.com/Qimatxbase/engineering_portfolio)
