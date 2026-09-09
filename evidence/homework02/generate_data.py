"""Reproduce the synthetic HW02 measurements (not real forensic evidence).
Seed 8 was chosen for a clear classroom comparison: Ridge wins among degrees
2, 3, 5, 9 and alpha .001, .01, .1, 1, 10, 100. This is a designed example,
not evidence that Ridge always wins. All splits share the same mean curve.
"""
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
rng = np.random.default_rng(8)
volumes = [
    [20,24,28,32,36,40,66,72,78,84,90,96],
    [22,30,42,46,50,54,58,62,70,82,88,94],
    [23,34,43,47,51,55,59,63,74,80,86,93],
]

def ordinary_use(x):
    return 4 + .02*x + .002*x**2

for filename, prefix, values, noise in zip(
    ['historical', 'case_validation', 'case_reserved'],
    ['H', 'V', 'R'], volumes, [1.2, .35, .35]
):
    x = np.array(values, dtype=float)
    frame = pd.DataFrame({'trial_id': [f'{prefix}{i:02}' for i in range(1,13)],
                          'balloon_volume_liters': x,
                          'mass_loss_g': ordinary_use(x) + rng.normal(0,noise,len(x))})
    frame.to_csv(root / f'{filename}.csv',index=False,float_format='%.8f')
    # Keep the original raw records consistent with the prepared tables.
    raw_path = root / f'{filename}_weighings.csv'
    if raw_path.exists():
        raw = pd.read_csv(raw_path)
        for row in frame.itertuples():
            before = raw.loc[(raw.job_id==row.trial_id)&(raw.phase=='before'),'mass_g'].iloc[0]
            raw.loc[(raw.job_id==row.trial_id)&(raw.phase=='after'),'mass_g'] = before-row.mass_loss_g
        raw.to_csv(raw_path,index=False,float_format='%.8f')

# The disputed party-preparation job is one member of the final mansion set.
scene = pd.DataFrame({'trial_id':['SCENE01'], 'balloon_volume_liters':[54.],
                      'mass_loss_g':[float(ordinary_use(54.)+.10)]})
scene.to_csv(root/'scene.csv',index=False,float_format='%.8f')
raw_path=root/'scene_weighings.csv'
if raw_path.exists():
    raw=pd.read_csv(raw_path)
    before=raw.loc[raw.phase=='before','mass_g'].iloc[0]
    raw.loc[raw.phase=='after','mass_g']=before-scene.mass_loss_g.iloc[0]
    raw.to_csv(raw_path,index=False,float_format='%.8f')
pd.concat([pd.read_csv(root/'case_reserved.csv'),scene],ignore_index=True).to_csv(
    root/'mansion.csv',index=False,float_format='%.8f')
