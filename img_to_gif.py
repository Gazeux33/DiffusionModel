import glob
from PIL import Image


def make_gif(frame_folder):
    frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.jpg")]
    frame_one = frames[0]
    frame_one.save("results/cat1.gif", format="GIF", append_images=frames,
                   save_all=True, duration=100, loop=1)


if __name__ == "__main__":
    make_gif("results/image_gif1")