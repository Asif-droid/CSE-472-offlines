import matplotlib.pyplot as plt
import imageio

def create_gif():
    images = []
    
    for i in range(50):
        # Create your plot here
        plt.plot([0, i], [0, i])
        plt.title(f'Iteration {i+1}')
        
        # Save the plot as an image
        filename = f'plot_{i+1}.png'
        plt.savefig(filename)
        plt.close()
        
        # Append the image to the list
        images.append(imageio.imread(filename))
    
    # Save the list of images as a GIF file
    imageio.mimsave('animation.gif', images, duration=0.5)
