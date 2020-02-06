import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from tqdm import tqdm

roc_dir = '../plots/roc/'
files = [f for f in os.listdir(roc_dir) if f.endswith("h5")]

tpr = []
fpr = []
roc_auc = []
models = []

for(i, name) in zip(range(len(files)), tqdm(files)):
    print("Reading file", i, name)
    data = pd.read_hdf(roc_dir + name)
    tpr.append(data.loc['tpr'].values)
    fpr.append(data.loc['fpr'].values)
    roc_auc.append(auc(fpr[i], tpr[i]))
    model = name.rsplit('_',1)[0]
    models.append(model)

# plot ROC curves

plt.figure()
for i in range(len(tpr)):
    plt.plot(fpr[i], tpr[i], color='C'+str(i), lw=1.5, label=models[i]+' (AUC = %0.2f)' % roc_auc[i])
plt.plot([0, 1], [0, 1], color='black', lw=1.5, linestyle='--')
plt.plot([0, 1], [1, 1], color='black', lw=0.7, linestyle=':')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC comparison')
plt.legend(loc="lower right", fontsize=9.3)
plt.savefig(roc_dir + 'roc_comparison.pdf', format='pdf')
plt.show()

