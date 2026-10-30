"""Independent, fresh 5 cm curriculum with a strict per-stage training cap."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import random
import shutil
from datetime import datetime
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path('result_5cm/.matplotlib').resolve()))
from gym_pybullet_drones.examples.curriculum_navigate import (
    ROOT, STAGES, History, PPO, checkpoint, save_json, passed, configure,
    torch, np, make_vec_env, VecFrameStack, ObservationType, ActionType, NavigateExtractor)
from gym_pybullet_drones.envs.CenterGoalAviary import CenterGoalAviary


def make_env(stage, count=1, monitor=None):
    env=make_vec_env(CenterGoalAviary,n_envs=count,seed=0,monitor_dir=monitor,
                     env_kwargs=dict(difficulty=stage,gui=False,
                                     obs=ObservationType.ALL,act=ActionType.VEL))
    return VecFrameStack(env,4,channels_order='first')


class CappedHistory(History):
    def __init__(self,directory,env,origin,interval,limit):
        super().__init__(directory,env,origin,0,interval)
        self.limit=limit
        self.last_evaluation=None

    def record(self):
        super().record()
        self.last_evaluation=self.model.num_timesteps-self.origin

    def _on_step(self):
        steps=self.num_timesteps-self.origin
        if steps>self.limit:
            raise RuntimeError('Training exceeded the per-stage limit')
        keep_training=super()._on_step()
        if not keep_training:
            return False
        if steps==self.limit:
            if self.last_evaluation!=steps:
                self.record()
            return False
        return True


def train_stage(output,stage,previous,args):
    directory=output/stage;directory.mkdir()
    train=make_env(stage,args.n_envs,str(directory/'monitor'))
    evaluation=make_env(stage)
    origin=0
    model=PPO('MultiInputPolicy',train,device='cpu',
        learning_rate=1e-5 if stage=='hard' else 1e-4,
        n_steps=args.rollout_steps,batch_size=args.batch_size,n_epochs=10,
        clip_range=.15,target_kl=.015,ent_coef=.005,vf_coef=.05,verbose=0,
        policy_kwargs=dict(features_extractor_class=NavigateExtractor,
            features_extractor_kwargs=dict(cnn_output_dim=32),share_features_extractor=False,net_arch=[64,64]))
    if previous:
        assert previous.resolve().is_relative_to(output)
        old=PPO.load(previous,device='cpu')
        model.policy.load_state_dict(old.policy.state_dict(),strict=True)
        model.policy.log_std.data.fill_(-1.2)
        model.num_timesteps=origin=old.num_timesteps
    save_json(directory/'run_config.json',dict(stage=stage,arrival_threshold=.05,max_steps=args.max_steps,
        source_timesteps=origin,source_model=str(previous) if previous else None,
        source_sha256=hashlib.sha256(previous.read_bytes()).hexdigest() if previous else None,
        initialization='previous passed stage' if previous else 'random weights, seed 42'))
    checkpoint(model,directory/'initial_model.zip')
    model.set_logger(configure(str(directory),['csv','tensorboard']))
    save_json(output/'status.json',dict(state='training',stage=stage,pid=os.getpid(),max_steps_per_stage=args.max_steps))
    save_json(directory/'status.json',dict(state='evaluating_baseline',stage_steps=0,arrival_threshold=.05))
    callback=CappedHistory(directory,evaluation,origin,args.eval_interval,args.max_steps)
    callback.init_callback(model)
    try:
        callback.record()
        if not callback.gate:
            model.learn(total_timesteps=args.max_steps,reset_num_timesteps=False,
                        callback=callback,log_interval=1)
        steps=model.num_timesteps-origin
        assert steps<=args.max_steps
        if not callback.gate and callback.last_evaluation!=steps:
            callback.record()
        checkpoint(model,directory/'latest_model.zip')
        checkpoint(model,directory/'final_model.zip')
        if callback.validation is not None:
            save_json(directory/'validation.json',callback.validation)
        state='complete' if callback.gate else 'stopped_limit'
        save_json(directory/'status.json',dict(state=state,gate_passed=callback.gate,
            arrival_threshold=.05,stage_steps=steps,max_steps=args.max_steps,
            stop_reason='100/100 screening and independent validation' if callback.gate else '600000-step cap reached without passing' if args.max_steps==600000 else 'step cap reached without passing'))
        return callback.gate
    except BaseException:
        checkpoint(model,directory/'interrupted_model.zip')
        save_json(directory/'status.json',dict(state='interrupted',stage_steps=model.num_timesteps-origin))
        save_json(output/'status.json',dict(state='interrupted',stage=stage))
        raise
    finally:
        train.close();evaluation.close()


def run(args):
    import gym_pybullet_drones
    assert Path(gym_pybullet_drones.__file__).resolve().parent==ROOT/'gym_pybullet_drones'
    if not 0<args.max_steps<=600000 or args.max_steps%args.n_envs:
        raise ValueError('Use at most 600000 steps, divisible by the environment count')
    torch.set_num_threads(1)
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    output=Path(args.output).resolve()
    if any((output/stage).exists() for stage in STAGES):
        raise FileExistsError('Use a new output directory; old stage data will not be overwritten')
    output.mkdir(parents=True,exist_ok=True)
    lock=(output/'training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save_json(output/'run_config.json',dict(vars(args),steps=[args.max_steps]*3,
        pid=os.getpid(),repository=str(ROOT),arrival_threshold=.05,
        started_at=datetime.now().astimezone().isoformat(),initialization='fresh random weights',
        transition='100/100 collision-free screening + independent 100/100 validation',
        on_limit_failure='stop entire curriculum; do not advance'))
    save_json(output/'versions.json',{name:importlib.metadata.version(name) for name in
        ['torch','numpy','pybullet','gymnasium','stable-baselines3']})
    snapshot=output/'source';snapshot.mkdir()
    for relative in ['envs/NavigateAviary.py','envs/CenterGoalAviary.py','utils/NavigateExtractor.py',
                     'control/DSLPIDControl.py','examples/curriculum_navigate.py','examples/train_center_curriculum.py']:
        shutil.copy2(ROOT/'gym_pybullet_drones'/relative,snapshot/Path(relative).name)
    save_json(output/'original_env.json',dict(path=str(ROOT/'gym_pybullet_drones/envs/NavigateAviary.py'),
        sha256=hashlib.sha256((ROOT/'gym_pybullet_drones/envs/NavigateAviary.py').read_bytes()).hexdigest(),
        geometry='original reset geometry',changes='subclass overrides only arrival radius to 0.05m'))
    previous=None
    for stage in STAGES:
        if not train_stage(output,stage,previous,args):
            save_json(output/'status.json',dict(state='stopped_limit',stage=stage,gate_passed=False,
                max_steps_per_stage=args.max_steps,reason='Stage did not pass within cap; later stages not started'))
            return
        previous=output/stage/'best_model.zip'
    save_json(output/'status.json',dict(state='complete',gate_passed=True,stages=STAGES))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',default='result_5cm')
    parser.add_argument('--max-steps',type=int,default=600000)
    parser.add_argument('--n-envs',type=int,default=8)
    parser.add_argument('--eval-interval',type=int,default=10000)
    parser.add_argument('--rollout-steps',type=int,default=2048)
    parser.add_argument('--batch-size',type=int,default=1024)
    args=parser.parse_args()
    if not 0<args.max_steps<=600000:parser.error('max-steps must be between 1 and 600000')
    if args.n_envs<=0 or args.max_steps%args.n_envs:parser.error('max-steps must be divisible by n-envs')
    if min(args.eval_interval,args.rollout_steps,args.batch_size)<=0:parser.error('Sizes must be positive')
    run(args)
