import argparse
import cv2
import os


def show_images(folder_name = 'raw_images'):
    for file_name in os.listdir(folder_name):
        img = cv2.imread(folder_name + '/' + file_name)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL) #resize large images
        cv2.imshow('image', img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    """
    Show images from --folder_name argument
    """
    parser = argparse.ArgumentParser(description='Show images from --folder_name', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--folder_name', default='raw_images', help='Directory with images to show')
    args, other_args = parser.parse_known_args()

    show_images(args.folder_name)