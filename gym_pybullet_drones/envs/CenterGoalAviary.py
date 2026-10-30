"""A 5 cm arrival criterion on the original navigation environment."""
from gym_pybullet_drones.envs.NavigateAviary import NavigateAviary


class CenterGoalAviary(NavigateAviary):
    ARRIVAL_RADIUS = 0.05

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._geometry_threshold = self.ARRIVAL_THRESHOLD
        self.ARRIVAL_THRESHOLD = self.ARRIVAL_RADIUS

    def reset(self, seed=None, options=None):
        # Original reset uses ARRIVAL_THRESHOLD to place the target. Preserve
        # that geometry while changing only the success/terminal-bonus radius.
        self.ARRIVAL_THRESHOLD = getattr(self, '_geometry_threshold', self.ARRIVAL_THRESHOLD)
        try:
            observation, _ = super().reset(seed=seed, options=options)
        finally:
            self.ARRIVAL_THRESHOLD = self.ARRIVAL_RADIUS
        return observation, self._computeInfo()
