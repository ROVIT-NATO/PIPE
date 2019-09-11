import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['toolbar'] = 'None'
plt.ion()

class get_window:
    def __init__(self):
        self.fig = None
        self.columns = None
        self.rows = None
        self.sub_plot_list = []

    def create_plot(self, InFigureSize=(8, 8), InColumns=2, InRows=2, InTitle='Kingston University'):
        self.fig = plt.figure(figsize=InFigureSize)
        self.fig = plt.gcf()
        self.fig.canvas.set_window_title(InTitle)
        self.columns = InColumns
        self.rows = InRows

    def add_sub_plot(self, InImage, InPosition, InTitle, IsVisibleAxis=False):
        plot = self.fig.add_subplot(self.rows, self.columns, InPosition)
        plot.set_title(InTitle)
        plot.axes.get_xaxis().set_visible(IsVisibleAxis)
        plot.axes.get_yaxis().set_visible(IsVisibleAxis)
        self.sub_plot_list.append([plot, InImage])
        plt.imshow(InImage)

    def add_text(self, InText, InFontSize=12, InXPos=1, InYPos=1, InColor='black', Infontweight='bold',
                 InAlighnment='left'):
        # plt.text(InXPos, InYPos, InText, fontsize=InFontSize, style='oblique', ha='center', , wrap=True)
        plt.text(InXPos, InYPos, InText, color=InColor, fontsize=InFontSize, ha=InAlighnment, va='bottom',
                 fontweight=Infontweight)

    def show(self):
        plt.draw()
        plt.pause(0.0001)
        plt.clf()

# img = np.random.randint(10, size=(256, 256))
# window = get_window()
# window.create_plot(InFigureSize=(8,8), InColumns=2, InRows=2)
# window.add_sub_plot(img, 1, 'Drone View')
# window.show()
