# Notebook sources

Descriptive filenames replace working titles; the PPO training/evaluation experiments remain distinct.

| Original | Current |
| --- | --- |
| `example_training.ipynb` | `notebooks/ppo_navigation_training.ipynb` |
| `example_training_mimic_uts.ipynb` | `notebooks/ppo_staged_training.ipynb` |
| `result.ipynb` | `notebooks/ppo_curriculum_evaluation.ipynb` |
| `result_5cm.ipynb` | `notebooks/ppo_precision_landing_evaluation.ipynb` |

The plotting notebooks now locate the repository from its package and plotting script, rather than requiring the old checkout name. Empty cells, an out-of-order hardware shutdown and a command for an absent hardware helper were removed from the training notebooks. Saved execution outputs are excluded.
