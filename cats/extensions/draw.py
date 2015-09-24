import numpy as np
from skimage import io, draw, exposure
from colorsys import hls_to_rgb
from random import randrange


def draw_particles(self, output=None, spots=None, tracks=True, rescale=True):
    """
    Draw the tracks and/or spots onto the images of the dataset as RGB images.

    Argument:
        output: the directory in which to write the files. If None, returns the images as a list of arrays.
        spots: if True, surrounds the spots on the images with a red circle. If None, only show spots if no tracks are available.
        tracks: if present, surround each of the spots in the tracks on the images with a colored circle. All spots within the same track will have the same color.
        rescale: adjust intensity levels to the detected content
    """
    if output is None:
        images = list()

    if spots is None and self.tracks is None:
        spots = True

    # Separate tracks per image
    if tracks is True and self.tracks is not None:
        # Make colors
        colors = [np.array(hls_to_rgb(randrange(0, 360) / 360, randrange(20, 80, 1) / 100, randrange(20, 80, 10) / 100)) * 255 for i in self.tracks]
        frames = [[] for f in range(self.source.length)]
        for track, color in zip(self.get_tracks(), colors):
            sigma = int(np.mean([s['s'] for s in track]))
            for s in track:
                spot = (int(s['y']), int(s['x']), sigma, color)
                frames[s['t']].append(spot)

    # Calculate intensity range:
    if rescale == True:
        if tracks is True and self.tracks is not None:
            i = [i for t in self.get_tracks('i') for i in t]
        elif spots is True and self.spots is not None:
            i = [s['i'] for s in self.spots]
        if len(i) != 0:
            m, s = np.mean(i), np.std(i)
            scale = (m - 3 * s, m + 3 * s)
        else:
            scale = False

    shape = self.source.dimensions[::-1]
    for t, image in enumerate(self.source.read()):
        # Prepare the output image (a 8bits RGB image)
        ni = np.zeros([3] + list(self.source.dimensions[::-1]), dtype=np.uint8)

        if rescale is True and scale is not False:
            image = exposure.rescale_intensity(image, in_range=scale)

        ni[..., :] = image / image.max() * 255
        ni = ni.transpose(1, 2, 0)

        # Mark spots
        if spots is True and self.spots is not None:
            for s in (self.spots[i] for i in np.where(self.spots['t'] == t)[0]):
                area = draw.circle_perimeter(int(s['y']), int(s['x']), int(s['s']), shape=shape)
                ni[area[0], area[1], :] = (255, 0, 0)

        # Mark tracks
        if tracks is True and self.tracks is not None:
            for s in frames[t]:
                area = draw.circle_perimeter(s[0], s[1], s[2], shape=shape)
                ni[area[0], area[1], :] = s[3]

        # Show/save image
        if output is None:
            images.append(ni)
        else:
            io.imsave("{0}/{1}.tif".format(output, t), ni)
    return images if output is None else None

__extension__ = {'Dataset': {'draw_particles': draw_particles}}
