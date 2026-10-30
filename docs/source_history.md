# Source history

The published repository begins with a cleaned portfolio snapshot.

Original source repositories:

- https://github.com/Quang0309/AutoPilotDrone
- https://github.com/learnsyslab/gym-pybullet-drones

Original local Git databases, branch states and recovery copies are retained outside the public repositories in the workspace .maintenance/ directory.

## PPO consolidation

PPO is the only active navigation algorithm in this portfolio. The separate TD3 repository and its unique source/history were preserved in a private local recovery bundle before retirement. Shared source was compared: the PPO environment retains controller resets, and its feature extractor retains nonzero initialisation for gradient flow. TD3-specific replay/training code is not presented as PPO work.
