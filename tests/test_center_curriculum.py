import json
import zipfile
from argparse import Namespace
import numpy as np
from gym_pybullet_drones.envs.NavigateAviary import NavigateAviary
from gym_pybullet_drones.envs.CenterGoalAviary import CenterGoalAviary
from gym_pybullet_drones.examples import curriculum_navigate as original
from gym_pybullet_drones.examples import train_center_curriculum as runner


def test_goal_changes_only_arrival_not_original_geometry(monkeypatch):
    standard=NavigateAviary(difficulty='hard')
    center=CenterGoalAviary(difficulty='hard')
    try:
        np.random.seed(12345);standard.reset(seed=12345)
        np.random.seed(12345);center.reset(seed=12345)
        np.testing.assert_array_equal(standard.TARGET_POS,center.TARGET_POS)
        np.testing.assert_array_equal(standard.INIT_XYZS,center.INIT_XYZS)
        assert standard.ARRIVAL_THRESHOLD==.33
        assert center.ARRIVAL_THRESHOLD==.05
        state=np.zeros(20)
        monkeypatch.setattr(center,'_getDroneStateVector',lambda _:state)
        monkeypatch.setattr(center,'_check_collision',lambda:False)
        for offset,success in [([.049,0,0],True),([.051,0,0],False),([0,0,.051],False),([.04,.04,0],False)]:
            state[:3]=center.TARGET_POS+offset
            assert center._computeTerminated()==success
            assert center._computeInfo()['success']==success
    finally:standard.close();center.close()


def test_cap_stops_real_ppo_mid_rollout_and_does_not_advance(monkeypatch,tmp_path):
    def fail(model,env,seeds):
        return dict(episodes=100,success_rate=0.,collision_rate=0.,mean_reward=0.,mean_length=10.,
                    results=[dict(success=False,collision=False,reward=0.,length=10) for _ in range(100)])
    monkeypatch.setattr(original,'evaluate',fail)
    monkeypatch.setattr(runner.CappedHistory,'refresh_charts',lambda self:None)
    args=Namespace(output=str(tmp_path/'run'),max_steps=40,n_envs=8,eval_interval=24,
                   rollout_steps=32,batch_size=16)
    runner.run(args)
    output=tmp_path/'run'
    status=json.loads((output/'status.json').read_text())
    assert status['state']=='stopped_limit'
    assert not (output/'medium').exists() and not (output/'hard').exists()
    assert json.loads((output/'easy/status.json').read_text())['stage_steps']==40
    with zipfile.ZipFile(output/'easy/final_model.zip') as z:
        assert json.loads(z.read('data'))['num_timesteps']==40
    rows=[json.loads(line) for line in (output/'easy/evaluations.jsonl').read_text().splitlines()]
    assert [row['stage_steps'] for row in rows]==[0,24,40]
    assert json.loads((output/'easy/run_config.json').read_text())['source_model'] is None
