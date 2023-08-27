import numpy as np
import matplotlib.pyplot as plt


def _scale_data(data, ranges):
    """scales data[1:] to ranges[0]"""
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        assert (y1 <= d <= y2) or (y2 <= d <= y1)
    x1, x2 = ranges[0]
    d = data[0]
    sdata = [d]
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        if y1 > y2:
            d = _invert(d, (y1, y2))
            y1, y2 = y2, y1
        sdata.append((d-y1) / (y2-y1)
                     * (x2 - x1) + x1)
    return sdata

class ComplexRadar():
    def __init__(self, fig, variables, ranges,
                 n_ordinate_levels=5):
        angles = np.arange(0, 360, 360./len(variables))
       
        axes = [fig.add_axes([0.1,0.1,0.9,0.9],polar=True,
                label = "axes{}".format(i))
                for i in range(len(variables))]
        l, text = axes[0].set_thetagrids(angles,
                                         labels=variables)
        [txt.set_rotation(angle-90) for txt, angle
             in zip(text, angles)]
        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)
        for i, ax in enumerate(axes):
            grid = np.linspace(*ranges[i],
                               num=n_ordinate_levels)
            gridlabel = ["{}".format(round(x,1))
                         for x in grid]
            gridlabel[0] = "" # clean up origin
            gridlabel[-1] = "" # clean up origin
            ax.set_rgrids(grid, labels=gridlabel,
                         angle=angles[i])
            #ax.spines["polar"].set_visible(False)
            ax.set_ylim(*ranges[i])
        # variables for plotting
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]
    def plot(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)
        self.ax.legend()
    def fill(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)
        self.ax.legend()



# example data
if __name__ == '__main__':
    variables = ("Var 1", "Var 2", "Var 3",
                 "Var 4", "Var 5", "Var 6",
                 "Var 7")
    data1 = (1.76, 1.1, 1.2, 4.4, 3.4, 86.8, 20)
    data2 = (2.1, 0.1, 1.7, 4, 3, 72, 20)
    ranges = [(0.1, 2.3), (0,1.5), (0,5),
              (1.7, 4.5), (1.5, 3.7), (70, 87), (10,100)]
               
    # plotting
    fig = plt.figure(figsize=(8, 8))
    
    radar = ComplexRadar(fig, variables, ranges)
    radar.plot(data1, ms=4, marker='o', label='bkg')
    #radar.fill(data1, alpha=0.2)
    
    radar.plot(data2, ms=4, marker='o', color='orange')
    radar.fill(data2, alpha=0.2, color='orange',label='signal')
    
    fig.savefig('plots/radar_plot.pdf', bbox_inches="tight")
