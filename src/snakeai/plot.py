import matplotlib
import matplotlib.pyplot as plt

plt.ion()


def plot(scores, mean_scores, save_path=None):
    plt.clf()
    plt.title("Training...")
    plt.xlabel("Number of Games")
    plt.ylabel("Score")
    plt.plot(scores, label="Score")
    plt.plot(mean_scores, label="Mean Score")
    plt.ylim(ymin=0)
    if scores:
        plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    if mean_scores:
        plt.text(len(mean_scores) - 1, mean_scores[-1], f"{mean_scores[-1]:.2f}")
    plt.legend(loc="upper left")
    plt.show(block=False)
    plt.pause(0.1)

    if save_path:
        plt.savefig(save_path)
