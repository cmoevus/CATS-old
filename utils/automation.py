"""Track datasets defined as dictionaries."""

def track_dataset(data):
    r"""Load data as explained in the given dict.

    Parameters
    ----------
    data: dict
        The dictionary explaining the data to be loaded. See the section *Notes* for explanations on how to write the dict.

    Returns
    -------
    dict
        A dict with datasets names as keys and structure identical to the `data` dict, containing the tracking results according to the parameters given.

    Notes
    -----

    Syntax of the input `data` dictionary.
    ++++++++++++++++++++++++++++++++++++++
    {
        # Merge datasets of different sources with different parameters
        'dataset': {
            'name': 'name',
            'dataset': {
                parameters
            }
            'dataset': {
                parameters
            }
            ...
            parameters
        }

        # Merge datasets of different sources with the same parameters
        'dataset': {
            'name': 'name',
            'images': ['/path/1', 'path/2', ...]
            parameters
        }
        parameters
    }

    - Datasets can be nested at will.
    - All unnamed, sibling datasets will be merged into the parent dataset, or as one dataset if some of their sibling datasets are named.
    - Datasets will be numbered in the order they appear in the output of the items() method, if they are not named and need to be.
    - Parameters are inherited from parent datasets and can be overridden.

    Standard parameters for the input data
    +++++++++++++++++++++++++++++++++++++++
    images : str
        location of the images (directory of file), relative to dir if dir is given.
    name : string, int
        name of the dataset
    dir : str
        parent path
    filter : func
        filtering function
    tracking : dict
        arguments to be passed to the tracking function
    t : {2,3-tuple, slice}
        Time ROI
    x : {2,3-tuple, slice}
        x-axis ROI
    y : {2,3-tuple, slice}
        y-axis ROI
    c : int, list
        channel(s) of interest
    barriers : array of {2,3-tuple, slice}
        position of the barriers. 2,3-tuples or slices with the first value being the position of the barrier and the last, that of the pedestal.
    test: int
        Only track the defined number of frames for each dataset, to test tracking parameters. Default: 0 (all frames)

    User parameters for the input data
    ++++++++++++++++++++++++++++++++++++
    Users can input their own parameters. These parameters will be appended as attributes to the "Contents
    framerate

    """
    sub, dataset = dict(), content.Particles()

    # Recursively add datasets
    for k, v in data.items():
        if 'dataset' in k:
            if 'name' in v:
                sub[v['name']] = track_dataset(v)
            else:
                d = track_dataset(v)
                if len(dataset) > 0:
                    dataset.extend(d)
                else:
                    dataset = d

    # Track
    if 'images' in data:
        # Get images
        I = sources.ROI(data['dir'] + data['images'], **dict((k, v) for k, v in data.items() if v is not None and k in ['x', 'y', 't', 'c']))

        # Test tracking parameters only?
        if data['test'] is not None and data['test'] != 0:
            I.t = (I.t.start, I.t.start + data['test'], 1)

        # Consider barriers?
        if data['barriers'] is not None:
            for b in data['barriers']:
                R = sources.ROI(I, x=b)
                p_b = detect.particles.stationary(R, **data['tracking'])
                if len(dataset) > 0:
                    dataset.extend(p_b)
                else:
                    dataset = p_b
        else:
            d = detect.particles.stationary(I, **data['tracking'])
            if len(dataset) > 0:
                dataset.extend(d)
            else:
                dataset = d

    # Filter
    if data['filter'] is not None:
        dataset = data['filter'](dataset)

    # Append parameters
    for k, v in data.items():
        if k not in ['images']:
            setattr(dataset, k, v)

    # Merge
    if len(sub) == 0:
        ret = dataset
    elif len(dataset) == 0:
        ret = sub
    else:
        sub[0] = dataset
        ret = sub

    return ret
