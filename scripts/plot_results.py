"""Policy and critic figures from actual saved curriculum histories."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

STAGES=['easy','medium','hard']
COLORS=['#247eb4','#ff7f0e','#2ca02c']

def load(path, default):
    return json.loads(path.read_text()) if path.exists() else default

def read_history(result, critic=False):
    data={stage:[] for stage in STAGES}
    for stage in STAGES:
        if critic:
            paths=[result/stage/'progress.csv',*sorted((result/stage/'segments').glob('*/progress.csv'))]
            for path in paths:
                if not path.exists():continue
                with path.open() as handle:
                    for row in csv.DictReader(handle):
                        keys=['time/total_timesteps','train/value_loss','train/explained_variance']
                        if all(row.get(key) for key in keys):
                            data[stage].append(dict(stage=stage,timesteps=int(float(row[keys[0]])),
                                                   first=float(row[keys[1]]),second=float(row[keys[2]])))
        else:
            path=result/stage/'evaluations.jsonl'
            lines=path.read_text().splitlines() if path.exists() else []
            for index,line in enumerate(lines):
                if not line.strip():continue
                try:row=json.loads(line)
                except json.JSONDecodeError:
                    if index==len(lines)-1:break
                    raise
                rewards=np.array([episode['reward'] for episode in row['results']])
                lengths=np.array([episode['length'] for episode in row['results']])
                data[stage].append(dict(stage=stage,timesteps=row['timesteps'],
                    first=float(rewards.mean()),second=float(lengths.mean()),
                    first_std=float(rewards.std()),second_std=float(lengths.std())))
        data[stage].sort(key=lambda row:row['timesteps'])
    return data

def plot(result=Path('result'), critic=False):
    result=Path(result);data=read_history(result,critic)
    config=load(result/'run_config.json',{})
    targets=config.get('steps',[500000,400000,300000])
    origin=data['easy'][0]['timesteps'] if data['easy'] else 0
    bounds=[0]
    for index,stage in enumerate(STAGES):
        stage_config=load(result/stage/'run_config.json',{})
        start=stage_config.get('source_timesteps',bounds[-1])
        next_config=load(result/STAGES[index+1]/'run_config.json',{}) if index<2 else {}
        end=next_config.get('source_timesteps',max(start+targets[index],max((r['timesteps'] for r in data[stage]),default=0)))
        bounds.append(end)
    bounds=[0]+[step-origin for step in bounds[1:]]
    fig,axes=plt.subplots(2,1,figsize=(15,8.5),sharex=True,gridspec_kw={'hspace':.32})
    fig.suptitle('PPO Critic Optimization Across Curriculum Stages' if critic else
                 'PPO Policy Evaluation Across Curriculum Stages',fontsize=17,fontweight='bold',y=.985)
    titles=['Critic Value Loss','Critic Explained Variance'] if critic else ['Evaluation Episodic Return','Evaluation Episode Length']
    for index,stage in enumerate(STAGES):
        rows=data[stage]
        for axis,title,metric in zip(axes,titles,['first','second']):
            axis.set_title(title,fontweight='bold',pad=12)
            axis.axvspan(bounds[index],bounds[index+1],color=COLORS[index],alpha=.065)
            if index:axis.axvline(bounds[index],color='#777',linestyle='--',linewidth=1)
            axis.grid(alpha=.2,linestyle=':')
            if rows:
                x=np.array([row['timesteps']-origin for row in rows])
                y=np.array([row[metric] for row in rows])
                if critic:
                    smooth=np.array([y[max(0,i-4):i+1].mean() for i in range(len(y))])
                    axis.scatter(x,y,color=COLORS[index],s=12,alpha=.25)
                    axis.plot(x,smooth,color=COLORS[index],linewidth=2,label=f'{stage.title()} (last: {y[-1]:.3f})')
                else:
                    std=np.array([row[metric+'_std'] for row in rows])
                    axis.plot(x,y,color=COLORS[index],linewidth=1.8,marker='o' if len(rows)==1 else None,
                              label=f'{stage.title()} (last: {y[-1]:.2f})')
                    if len(rows)>1:
                        lower=y-std if metric=='first' else np.maximum(0,y-std)
                        axis.fill_between(x,lower,y+std,color=COLORS[index],alpha=.17)
                    else:axis.errorbar(x,y,yerr=std,color=COLORS[index],capsize=4)
        label=f'Stage {index+1}: {stage.title()}\n'+(f'Last: {rows[-1]["first"]:.2f}' if rows else 'Awaiting data')
        axes[0].text(np.mean(bounds[index:index+2]),.96,label,ha='center',va='top',
                     transform=axes[0].get_xaxis_transform(),fontweight='bold',
                     bbox=dict(boxstyle='round,pad=.35',fc='white',ec=COLORS[index]))
    axes[0].set_ylabel('Value Loss (Log Scale)' if critic else 'Mean Evaluation Return')
    axes[1].set_ylabel('Explained Variance' if critic else 'Episode Length (Steps)')
    if critic:
        if any(row['first']>0 for rows in data.values() for row in rows):axes[0].set_yscale('log')
        axes[1].axhline(1,color='green',linestyle=':',label='Ideal (1.0)')
        axes[1].axhline(0,color='red',linestyle=':',label='Baseline (0.0)')
    else:
        axes[0].axhline(0,color='#888',linestyle=':')
        axes[1].set_ylim(bottom=0)
    for axis in axes:
        axis.set_xlim(0,max(bounds[-1],1)*1.02);axis.margins(y=.3)
        if axis.get_legend_handles_labels()[0]:axis.legend(fontsize=9)
    axes[1].set_xlabel('Timesteps Since First Recorded Easy Step')
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v/1e6:.1f}M' if v>=1e6 else f'{v/1000:.0f}k'))
    caption='Dots: training logs; lines: trailing five-point means.' if critic else 'Evaluation mean ±1 SD across 100 episodes; 24 control steps/s.'
    fig.text(.5,.02,caption+' Future stage spans are provisional.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.08,right=.985,bottom=.10,top=.91)
    output=result/'plots';output.mkdir(parents=True,exist_ok=True)
    name='curriculum_critic' if critic else 'curriculum_evaluation'
    for extension in ['png','svg','pdf']:fig.savefig(output/f'{name}.{extension}',dpi=180,facecolor='white')
    rows=[row for group in data.values() for row in group]
    if rows:
        with (output/f'{name}_history.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return fig

def refresh(result):
    for critic in [False,True]:plt.close(plot(result,critic))
