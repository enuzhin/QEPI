import os
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt

os.makedirs("images", exist_ok=True)
os.makedirs("data", exist_ok=True)


def use_paper_style():
    SMALL_SIZE = 10  # 8
    MEDIUM_SIZE = 12  # 10
    BIGGER_SIZE = 14  # 12
    plt.rcParams["font.family"] = ["Times New Roman"]
    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



def save_trainig_data(extent, data, name, x_label="Position", y_lebel="Velocity",
                      ticks=[0,3,6,9,12,15,18], xticklabels=None, yticklabels=None,
                      cmap='magma', norm_gamma=1, format="pdf", dpi=600):

    plt.figure()
    norm=colors.PowerNorm(gamma=norm_gamma,vmin=0,vmax=1)
    plt.imshow(data[:,::-1].T, norm = norm, cmap = cmap,extent = extent,aspect = 'auto')
    plt.xlabel(x_label)
    plt.ylabel(y_lebel)
    if yticklabels is not None:
        plt.yticks(range(len(yticklabels)), [str(v) for v in yticklabels])
    if xticklabels is not None:
        plt.xticks(range(len(xticklabels)), [str(v) for v in xticklabels])
    cbar = plt.colorbar(ticks=ticks)
    if ticks is not None:
        cbar.ax.set_yticklabels(ticks)
    plt.savefig("images/"+name + "."+format,dpi = dpi)
    plt.close()



def save_policy(X,V,data, name, x_label = "Position", y_lebel = "Velocity", cmap = 'PRGn', norm_gamma = 1,
              use_cbar = True, ticks = None,ticklabels = None, vmin = None, vmax = None,format = "svg",dpi = None):
    plt.figure()
    norm=colors.PowerNorm(gamma=norm_gamma,vmin=vmin,vmax=vmax)
    plt.imshow(data[:,::-1].T, cmap = cmap, norm = norm,extent = (X.min(),X.max(),V.min(),V.max()),aspect = 'auto')
    plt.xlabel(x_label)
    plt.ylabel(y_lebel)
    if use_cbar:
        cbar = plt.colorbar(ticks=ticks)
    if ticklabels is not None:
        cbar.ax.set_yticklabels(ticklabels)
    plt.savefig("images/"+name + "."+format,dpi = dpi)
    plt.close()


def save_value(X,V,data,name, x_label = "Position", y_lebel = "Velocity",ticks = None, cmap = 'magma', norm_gamma = 1,format = "pdf",dpi = 600):
    plt.figure()
    norm=colors.PowerNorm(gamma=norm_gamma)
    plt.imshow(data[:,::-1].T, cmap = cmap, norm = norm,extent = (X.min(),X.max(),V.min(),V.max()),aspect = 'auto')
    plt.xlabel(x_label)
    plt.ylabel(y_lebel)
    cbar = plt.colorbar(ticks=ticks)
    plt.savefig("images/"+name + "."+format,dpi = dpi)
    plt.close()

def plot(X,V,data, x_label = "Position", y_lebel = "Velocity", cmap = 'viridis'):
    plt.pcolor(X,V,data, cmap = cmap)
    plt.xlabel(x_label)
    plt.ylabel(y_lebel)
    plt.colorbar()
    plt.show()



def save_optimality_bars(freq, name, format="pdf", dpi=600):
    plt.figure()
    plt.bar(np.arange(len(freq)), freq, color="purple")
    plt.grid();
    plt.xlabel("Update step");
    plt.ylabel("Fraction of optimal policies")
    plt.xticks(np.arange(len(freq)))
    plt.savefig(f"images/{name}.{format}", dpi=dpi);
    plt.close()


def save_accuracy_curve(x, mean, std, reference=None, x_label="Number of samples",
                        name="accuracy", format="pdf", dpi=600):
    plt.figure()
    plt.loglog(x, mean, linestyle='--', marker='o', color="green", label="Standard")
    plt.fill_between(x, mean - std, mean + std, color="green", alpha=0.1)
    if reference is not None:
        m, s = reference
        plt.hlines(m, x[0], x[-1], color="purple", label="LeapHybrid (ref)")
        plt.fill_between(x, m - s, m + s, color="purple", alpha=0.1)
    plt.legend(); plt.grid()
    plt.ylim([1e-1, 1e2]); plt.xticks(x, [str(v) for v in x])
    plt.xlabel(x_label); plt.ylabel("Accuracy of solution (L2)")
    plt.savefig(f"images/{name}.{format}", dpi=dpi); plt.close()

