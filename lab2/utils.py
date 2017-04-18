import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (14.0, 8.0)

def costs_accuracies_plot(acc_train, acc_val, cost_train, cost_val, destfile):
    plt.subplot(1, 2, 1)
    plt.plot(cost_train, 'r-', label='Train')
    plt.plot(cost_val, 'b-', label='Validation')
    plt.title('Cost function')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid('on')

    plt.subplot(1, 2, 2)
    plt.plot(acc_train, label='Train')
    plt.plot(acc_val, label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid('on')

    plt.savefig(destfile)
    plt.clf()
    plt.close()
    return destfile


def show_plot(plot_file):
    plot_img = plt.imread(plot_file)
    plt.imshow(plot_img)
    plt.axis('off')
    plt.show()
