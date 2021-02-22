from pycocotools.coco import COCO
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.path as path
import matplotlib.patches as patches
import numpy as np
import seaborn as sns

validation_json = "PAA/datasets/coco/annotations/instances_val2017.json"
output_dir = "PAA/output/"
val_coco = COCO(validation_json)

sigma_per_level = defaultdict(list)
sigma_per_top = defaultdict(list)
stride_per_level = [8, 16, 32, 64, 128]
INF = 1e7

# get statistics from dataset

for k, v in val_coco.anns.items():
    img_id = v['image_id']
    img_w, img_h = val_coco.imgs[img_id]['width'] , val_coco.imgs[img_id]['height'] 
    min_length = min(img_w, img_h)
    ratio = 800 / min_length
    w, h = np.array(v['bbox'][2:]) * ratio
    D = np.sqrt(w ** 2 + h ** 2)
    curr_sigma = []
    
    for s in stride_per_level:
        sigma_per_level[s].append(np.log((D / s) ** 2) / 20 + 0.6)
        curr_sigma.append(np.log((D / s) ** 2) / 20 + 0.6)

    near_one_idx = np.argsort(np.abs(np.log(curr_sigma)))
    for i, idx in enumerate(near_one_idx):
        sigma_per_top[i].append(curr_sigma[idx])
    
# plot histogram about sigma

fig, ax = plt.subplots(5, 2, figsize=(8,15))

for i, s in enumerate(stride_per_level):

    level_ax = ax[i,0]
    top_ax = ax[i,1]

    st_point = 0.5
    end_point = 1.0
    data = np.array(sigma_per_level[s])
    ax_info = sns.histplot(ax=level_ax, data=data, bins=25, log_scale=True, kde=True)
    level_ax.axline((1.0,0.0), (1.0,1.0), c="red", lw=3.0, ls='--')
    level_ax.axline((end_point,0.0), (end_point,1.0), c="blue", lw=1.0, ls='--')
    level_ax.axline((st_point,0.0), (st_point,1.0), c="blue", lw=1.0, ls='--')
    _, y_end = level_ax.set_ylim(auto=True)
    cover = patches.Rectangle((st_point, 0.0), end_point - st_point, y_end, facecolor="green", edgecolor=None, alpha=0.3)
    level_ax.add_patch(cover)
    ratio = ((data > st_point) * (data < end_point)).sum() / len(data)
    level_ax.text(0.6, y_end/2, "{:.4f}".format(ratio), color="midnightblue")


    data = np.array(sigma_per_top[i])
    sns.histplot(ax=top_ax, data=sigma_per_top[i], bins=25, log_scale=True, kde=True)
    top_ax.axline((1.0,0.0), (1.0,1.0), c="red", lw=3.0, ls='--')
    top_ax.axline((end_point,0.0), (end_point,1.0), c="blue", lw=1.0, ls='--')
    top_ax.axline((st_point,0.0), (st_point,1.0), c="blue", lw=1.0, ls='--')
    _, y_end = top_ax.set_ylim(auto=True)
    cover = patches.Rectangle((st_point, 0.0), end_point - st_point, y_end, facecolor="green", edgecolor=None, alpha=0.3)
    top_ax.add_patch(cover)
    ratio = ((data > st_point) * (data < end_point)).sum() / len(data)
    top_ax.text(0.6, y_end/2, "{:.4f}".format(ratio), color="midnightblue")

    #top_ax.axline((0.0,0.0), (0.0,1.0), c="red", lw=10.0, ls='--')
    #sns.histplot(ax=top_ax, data=sigma_per_top, bins=25)
    """
    data = np.array(sigma_per_level[s])
    n, bins = np.histogram(data, 50)
    
    left = np.log(np.array(bins[:-1]))
    right = np.log(np.array(bins[1:]))
    bottom = np.zeros(len(left))
    top = np.log(bottom + n)

    XY = np.array([[left, left, right, right], [bottom, top, top, bottom]]).T

    # get the Path object
    barpath = path.Path.make_compound_path_from_polys(XY)

    # make a patch out of it
    patch = patches.PathPatch(barpath)
    level_ax.add_patch(patch)

    # update the view limits
    level_ax.set_xlim(left[0], right[-1])
    level_ax.set_ylim(bottom.min(), top.max())

    level_ax.set_title("sigma per stride - {}".format(s))
    level_ax.set_xlabel("sigma") 
    #level_ax.set_xscale("log")
    #level_ax.set_yscale("log")
    #level_ax.set_xticks(np.exp(left))
    #level_ax.set_yticks(np.exp(top))
    level_ax.set_ylabel("counts")
    level_ax.grid(True)

    data = np.array(sigma_per_top[i])
    n, bins = np.histogram(data, 50)
    
    left = np.array(bins[:-1])
    right = np.array(bins[1:])
    bottom = np.zeros(len(left))
    top = bottom + n

    XY = np.array([[left, left, right, right], [bottom, top, top, bottom]]).T

    # get the Path object
    barpath = path.Path.make_compound_path_from_polys(XY)

    # make a patch out of it
    patch = patches.PathPatch(barpath)
    top_ax.add_patch(patch)

    # update the view limits
    top_ax.set_xlim(left[0], right[-1])
    top_ax.set_ylim(bottom.min(), top.max())

    top_ax.set_title("sigma per top - {}".format(i))
    top_ax.set_xlabel("sigma") 
    #top_ax.set_xscale("log")
    #top_ax.set_yscale("log")
    top_ax.set_ylabel("counts")
    top_ax.grid(True)
    """

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig(output_dir + "plot.png", bbox_inches='tight')


