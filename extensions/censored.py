"""Define whether a Particle's end (dissociation, loss, etc.) is observed in the dataset.

Extension for the Particle object.

"""


def right_censored(self):
    """Define whether the particle is right-censored, i.e. that the end of the trajectory is not observed in this dataset because it was ended before the trajectory."""
    return self['t'].max() + self.max_blink >= len(self.source)


__extension__ = {'Particle': {'right_censored': right_censored}}
