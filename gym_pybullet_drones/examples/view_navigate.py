"""Slow overhead flights with FPV/depth, retaining each terminal scene."""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK','True')
import argparse
import random
import time
import cv2
import numpy as np
import pybullet as p
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.stacked_observations import StackedObservations
from gym_pybullet_drones.envs.NavigateAviary import NavigateAviary
from gym_pybullet_drones.examples.test_navigate import create_hud
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

WINDOW='Drone FPV / Depth'

def command(key):
    return {ord('q'):'quit',27:'quit',ord('n'):'next',ord('r'):'replay'}.get(key)

def run(args):
    if args.speed<=0:raise ValueError('speed must be positive')
    torch.set_num_threads(1)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    model=PPO.load(args.model,device='cpu')
    env=NavigateAviary(difficulty=args.difficulty,gui=not args.headless,
                       obs=ObservationType.ALL,act=ActionType.VEL)
    window=f'{args.difficulty.upper()} - {WINDOW} - N: next / R: replay / Q: close'
    stack=StackedObservations(1,4,env.observation_space,channels_order='first')
    try:
        obs,_=env.reset(seed=args.seed)
        obs=stack.reset({key:value[None] for key,value in obs.items()})
        if not args.headless:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI,0,physicsClientId=env.CLIENT)
            p.resetDebugVisualizerCamera(cameraDistance=1.95,cameraYaw=0,cameraPitch=-89.9,
                                         cameraTargetPosition=[0,0,0],physicsClientId=env.CLIENT)
        print('[VIEW] Stage:',args.difficulty,'model:',args.model,
              'seed:',args.seed,'speed:',args.speed,flush=True)
        start=time.perf_counter()
        for step in range(int(env.EPISODE_LEN_SEC*env.CTRL_FREQ)):
            action,_=model.predict(obs,deterministic=True)
            raw,reward,terminated,truncated,info=env.step(action[0])
            # No automatic reset: preserve the actual final state and camera.
            obs,_=stack.update({key:value[None] for key,value in raw.items()},np.array([False]),[info])
            done=terminated or truncated
            status='SUCCESS' if info['success'] and not info['collision'] else (
                str(info.get('truncation_reason') or 'FINISHED').upper() if done else 'NAVIGATING')
            if not args.headless:
                cv2.imshow(window,create_hud(env,action[0],step=step+1,ep=1,episodes=1,status=status))
                requested=command(cv2.waitKey(1)&255)
                if requested:return requested
                remaining=(step+1)*env.CTRL_TIMESTEP/args.speed-(time.perf_counter()-start)
                if remaining>0:time.sleep(remaining)
            if done:
                print('[RESULT]',args.difficulty,status,'steps:',step+1,'distance:',info['distance'],
                      'collision:',info['collision'],flush=True)
                break
        if not args.headless:
            print('[VIEW]',args.difficulty,'final scene held. Camera window: N next, R replay, Q/Esc close.',flush=True)
            while p.isConnected(env.CLIENT):
                requested=command(cv2.waitKey(50)&255)
                if requested:return requested
                if cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:return 'quit'
    finally:
        if p.isConnected(env.CLIENT):env.close()
        if not args.headless:cv2.destroyAllWindows()

if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--model',help='Checkpoint; defaults to result/<difficulty>/best_model.zip')
    parser.add_argument('--difficulty',choices=['easy','medium','hard'],default='hard')
    parser.add_argument('--stages',nargs='+',choices=['easy','medium','hard'],
                        help='Show stages in sequence, retaining each scene until N is pressed')
    parser.add_argument('--speed',type=float,default=.15)
    parser.add_argument('--seed',type=int,default=2026102321)
    parser.add_argument('--headless',action='store_true')
    args=parser.parse_args()
    if args.stages and args.model:parser.error('Use per-stage default models with --stages')
    selected_model=args.model
    for stage in args.stages or [args.difficulty]:
        args.difficulty=stage
        args.model=selected_model or f'result/{stage}/best_model.zip'
        while True:
            result=run(args)
            if result=='quit':raise SystemExit(0)
            if result!='replay':break
