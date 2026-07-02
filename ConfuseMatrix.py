import os
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(confusion_matrix, classes, savingPath, name):
    fontSize = 16

    num_classes = len(classes)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define colormap
    cmap = plt.cm.get_cmap('Blues')  # 使用蓝色色彩映射方案

    # Plot the confusion matrix
    im = ax.imshow(confusion_matrix, cmap=cmap)

    # Set ticks
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))

    # Set labels
    ax.set_xticklabels(classes, fontsize=fontSize-2)
    ax.set_yticklabels(classes, fontsize=fontSize-2)

    # set x\y name
    ax.set_xlabel('Predict Label', fontsize=fontSize)
    ax.set_ylabel('True Label', fontsize=fontSize)

    # Loop over data dimensions to create text annotations
    for i in range(num_classes):
        for j in range(num_classes):
            # Get the value of each cell in the confusion matrix
            cell_value = confusion_matrix[i, j]

            # Set the text color based on the contrast with the background color
            text_color = "black" if cell_value < 500 else "white"

            # Set the annotation text
            annotation_text = int(cell_value)

            # Add the text annotation to the plot
            ax.text(j, i, annotation_text, ha="center", va="center", color=text_color, fontweight='bold', fontsize=fontSize)  # 加粗字体显示


    # Save the figure
    #plt.savefig(savingPath + 'confusionMatrix_{}.png'.format(name), bbox_inches='tight')