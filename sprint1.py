from keras.datasets import mnist
import matplotlib.pyplot as plt
from collections import Counter

(x_train, y_train), (x_test, y_test) = mnist.load_data()


def least_common_digit(x_set, y_set):
    '''
       Input: x_set, the x values of the dataset and y_set, the y values of the  dataset
       Expected Output: The image from the x set of the least common digit.
    '''
    least_common = collections.Counter(y_set).most_common()[-1]
    for i,num in enumerate(y_set):
        if num == least_common:
            return x_set[i]

lc_train = least_common_digit(x_train, y_train)
lc_test = least_common_digit(x_test, y_test)


def most_common_digit(x_set, y_set):
    '''
       Input: x_set, the x values of the dataset and y_set, the y values of the  dataset
       Expected Output: The image from the x set of the most common digit.
    '''
    least_common = collections.Counter(y_set).most_common()[0]
    for i,num in enumerate(y_set):
        if num == most_common:
            return x_set[i]

    
mc_train = most_common_digit(x_train, y_train)
mc_test = most_common_digit(x_test, y_test)

def plot_two(im1, title1, im2, title2):
    '''
        Input: im1, a matrix representing a grayscale image and title1 a string,im2 a matrix representing 
        a grayscale image and title2 a string
        Expected Output: A tuple (fig, ax) representing a generated figure from matplotlib and two subplots 
        ready to display the inputed images with the given titles
    '''

    fig, ax = plt.subplots(2,2)
    ax[0,0].imshow(im1)
    ax[0,1].imshow(im2)
    ax[0, 0].set_title(title1)
    ax[0,1].set_title(title2)
    x = [0,10,20]
    y = [25,20,15,10,5,0]
    ax[0, 0].plot(x, y)
    ax[0, 1].plot(x, y)

    return (fig, ax)

plot_two(lc_train, 'Least Common Train', lc_test, 'Least Common Test'), plot_two(mc_train, 'Most Common Train', mc_test, 'Most Common Test')
plt.show()

