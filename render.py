import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

class RenderHeatMap:
    def __init__(self):
        self.frame = 0

    def renderframe(self, q):
        plt.figure()
        ax = sns.heatmap(q)
        fig = ax.get_figure()
        fig.savefig(f"rendered_animation/{self.frame:03d}.png")
        fig.clear()
        self.frame += 1