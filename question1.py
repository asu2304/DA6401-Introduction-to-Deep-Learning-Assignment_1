fashion_mnist_labels = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

plotted, plot_no = [], 1
plt.figure(figsize=(4, 2))

# Initialize W&B run
wandb.init(project="backprop-from-scratch", name="plot_unique_labels")
for index, label in enumerate(train_labels): 

    if len(plotted) == len(set(train_labels)):
        break
        
    if label not in plotted: 
        plt.subplot(2, 5, plot_no)
        plt.imshow(train_images[index], cmap='grey')
        plotted.append(label)
        plt.title(f'{fashion_mnist_labels[label]}', fontsize=8)
        plt.axis("off")
        plot_no += 1

# getting the final figure
fig = plt.gcf()

#loggint the figure of wandb
wandb.log({"unique_labels_plot": wandb.Image(fig)})

plt.show()
wandb.finish()