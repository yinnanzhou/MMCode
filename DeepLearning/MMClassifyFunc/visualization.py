import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def visualize_results(trainer, save_path=""):
    """
    Visualize the results of the training using confusion matrix, loss curves, and accuracy curves.

    Parameters:
    trainer: An object that contains the ground_truth, predictions, losses, and acces attributes.
    """
    # 绘制混淆矩阵
    cm = confusion_matrix(
        [x for x in trainer.ground_truth], [x for x in trainer.predictions]
    )
    plt.figure(figsize=(30, 20))

    plt.subplot(2, 1, 1)
    sns.heatmap(
        cm.astype("float") / cm.sum(axis=1)[:, np.newaxis],
        annot=True,
        fmt=".1f",
        cmap="Blues",
        cbar=False,
    )
    plt.title("Classification accuracy: {:.2%}".format(trainer.acces["test"][-1]))
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    # 获取损失和准确率
    train_losses = trainer.losses["train"]
    test_losses = trainer.losses["test"]
    train_acces = trainer.acces["train"]
    test_acces = trainer.acces["test"]

    # 绘制损失曲线
    plt.subplot(4, 1, 3)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(test_losses) + 1), test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss")
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(4, 1, 4)
    plt.plot(range(1, len(train_acces) + 1), train_acces, label="Train Accuracy")
    plt.plot(range(1, len(test_acces) + 1), test_acces, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Test Accuracy")
    plt.legend()

    if save_path == "":
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()


def visualize_predict(all_labels, all_preds, save_path=""):
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Calculate and print accuracy
    accuracy = np.trace(cm)/cm.sum()


    plt.figure(figsize=(30, 20))

    sns.heatmap(
        cm.astype("float") / cm.sum(axis=1)[:, np.newaxis],
        annot=True,
        fmt=".1f",
        cmap="Blues",
        cbar=False,
    )
    plt.title("Predict accuracy: {:.2%}".format(accuracy))
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    if save_path == "":
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()
