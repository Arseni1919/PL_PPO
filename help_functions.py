from CONSTANTS import *


def plot_(title, data):
    fig = px.line(data, title=title)
    fig.show()
    # plt.plot(data)
    # plt.title(title)
    # plt.show()
