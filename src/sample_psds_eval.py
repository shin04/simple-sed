
import glob
import os.path as osp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from psds_eval import PSDSEval, plot_psd_roc, plot_per_class_psd_roc

groundtruth = pd.read_csv('../meta/valid_meta_strong.csv')
metadata = pd.read_csv('../meta/valid_meta_duration.csv')

psds_eval = PSDSEval(ground_truth=groundtruth, metadata=metadata)


# in case this cell is executed many times, let's clean the PSDSEval
psds_eval.clear_all_operating_points()

print(f"Adding Operating Point {1:02d}/50", end="\r")
threshold = 0.5
det = pd.read_csv('../pred.csv')
info = {"name": f"Op {1:02d}", "threshold": threshold}
psds_eval.add_operating_point(det, info=info)

# compute the PSDS of the system represented by its operating points
psds = psds_eval.psds(max_efpr=100)

# plot the PSD-ROC and corresponding PSD-Score
plot_psd_roc(psds)
plt.savefig('../psds.png')
