from types import SimpleNamespace
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from gym_pybullet_drones.envs.NavigateAviary import NavigateAviary
from gym_pybullet_drones.utils.NavigateExtractor import NavigateExtractor
from gym_pybullet_drones.examples import curriculum_navigate as trainer


def score(successes, collision=False):
    return dict(episodes=100,success_rate=successes/100,collision_rate=.01 if collision else 0.,
                mean_reward=1.,results=[dict(success=i<successes,collision=collision and i==0) for i in range(100)])


def test_transition_requires_target_steps_and_two_perfect_batches(monkeypatch,tmp_path):
    results=iter([score(100),score(99),score(100),score(99),score(100),score(100)])
    calls=[];saved=[]
    def evaluate(model,env,seeds):
        calls.append(list(seeds));return next(results)
    monkeypatch.setattr(trainer,'evaluate',evaluate)
    monkeypatch.setattr(trainer,'checkpoint',lambda model,path:saved.append(path.name))
    callback=trainer.History(tmp_path,None,0,500000,20000)
    monkeypatch.setattr(callback,'refresh_charts',lambda:None)
    callback.model=SimpleNamespace(num_timesteps=40000)
    callback.record()
    assert not callback.gate and len(calls)==1  # No early transition before target.
    callback.model.num_timesteps=500000
    callback.record()
    assert not callback.gate
    callback.record()
    assert not callback.gate and 'passed_model.zip' not in saved
    callback.record()
    assert callback.gate and 'passed_model.zip' in saved
    assert saved.count('best_model.zip') == 2  # Initial best, then verified transfer policy.
    assert not set(calls[3]) & set(calls[5])
    assert not trainer.passed(score(100,collision=True))


def test_depth_gradient_and_pid_episode_reset(monkeypatch):
    torch.set_num_threads(1)
    env=NavigateAviary(difficulty='medium')
    stack=VecFrameStack(DummyVecEnv([lambda:env]),4,channels_order='first')
    try:
        calls=[];original=env.ctrl[0].reset
        def reset():calls.append(True);original()
        monkeypatch.setattr(env.ctrl[0],'reset',reset)
        env.reset();env.step(np.zeros(4));assert len(calls)==1
        env.reset();assert len(calls)==2
        extractor=NavigateExtractor(stack.observation_space)
        extractor({'depth':torch.rand(2,4,48,64),'state':torch.rand(2,60)}).sum().backward()
        assert extractor.cnn_features[0].weight.grad.abs().sum()>0
    finally:stack.close()

