import matplotlib.pyplot as plt


def visualization(title, xlabel, ylabel,
                  plt1_x, plt1_y, label1,
                  plt2_x, plt2_y, label2,
                  marker1='o', linestyle1='-', color1='b',
                  marker2='o', linestyle2='--', color2='r'):
    plt.figure(figsize=(14, 7))
    plt.plot(plt1_x, plt1_y, label=label1, marker=marker1, linestyle=linestyle1, color=color1)
    plt.plot(plt2_x, plt2_y, label=label2, marker=marker2, linestyle=linestyle2, color=color2)
    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.show()


def visualization_for_3(title, xlabel, ylabel,
                  plt1_x, plt1_y, label1,
                  plt2_x, plt2_y, label2,
                  plt3_x, plt3_y, label3):
    plt.figure(figsize=(14, 7))
    plt.plot(plt1_x, plt1_y, label=label1)
    plt.plot(plt2_x, plt2_y, label=label2)
    plt.plot(plt3_x, plt3_y, label=label3)
    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.show()
