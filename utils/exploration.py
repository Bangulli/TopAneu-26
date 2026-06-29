import seaborn as sns, matplotlib.pyplot as plt
import os, numpy as np, SimpleITK as sitk, pathlib as pl

dir = pl.Path("TopAneu-26/location_masks")

data = {i:0 for i in range(1,51)}
for file in os.listdir(dir):
    img = sitk.GetArrayFromImage(sitk.ReadImage(dir/file))
    for value in [v for v in np.unique(img).tolist() if v != 0]:
        data[value]+=1
        
fig = sns.barplot(data)
fig.tick_params(axis='x', labelrotation=90)
fig.get_figure().tight_layout()
plt.savefig("distribution_260626.png")