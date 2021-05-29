from PIL import Image
from matplotlib import image
from matplotlib import pyplot


def taskA():
    # load and show an image with Pillow

    # load the image
    image = Image.open('Assignment8/f1.jpg')

    # summarize some details about the image
    print(image.format)
    print(image.mode)
    print(image.size)

    # show the image
    image.show()


def taskB():
    # load and display an image with Matplotlib
    # load image as pixel array
    data = image.imread('Assignment8/f1.jpg')

    # summarize shape of the pixel array
    print(data.dtype)
    print(data.shape)

    # display the array of pixels as an image
    pyplot.imshow(data)
    pyplot.show()


def taskC():
    # Resize an image to a specific dimension
    # create a thumbnail of an image

    # load the image
    image = Image.open('Assignment8/f1.jpg')
    # report the size of the image
    print(image.size)
    # create a thumbnail and preserve aspect ratio
    image.thumbnail((100, 100))
    # report the size of the thumbnail
    print(image.size)

    # display the image thumbnail
    pyplot.imshow(image)
    pyplot.show()


if __name__ == '__main__':
    taskA()
