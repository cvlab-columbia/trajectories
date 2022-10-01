"""
Class to manage a dictionary of segments and trajectories
Every element in video_dicts contains a dictionary with
    - the tensor ('tensor'),
    - the space ('space': 'traj' if latent space, 'segment' if input space)
    - time it is representing ('time_label': 'past', 'future' for input space)
    - time steps in seconds ('time_steps', a tensor)
    - 'seg_len'. Number of actual steps in time_steps. Convenient because time steps may be padded
    - 'origin' (where the array comes from, which starts as '')
In the case of space == 'traj', the attributes 'time_label', 'time_steps' and 'seg_len', may not be None. In that case,
they do NOT represent information about the current trajectory tensor, but they give information about what segment
tensor they should be decoded to. They may be a list, meaning that they are decoded into more than one segment

The keys in the dictionary contain the same info as the dictionary. It is just clearer to also have it as a dictionary.
"""
from __future__ import annotations

from collections import OrderedDict, ChainMap

import numpy as np
import torch

from data.dataset import HumanKeypointsDataset


class TrajectoryDict:

    def __init__(self, dictionary):
        super().__init__()
        self.dictionary = dictionary

    @classmethod
    def create_from_batch(cls, batch, return_all=False) -> TrajectoryDict:
        """
        Create initial dict from inputs
        """

        dictionary = OrderedDict([(
            'seg_inp_p', {
                'tensor': batch['past'],
                'space': 'segment',
                'time_label': 'past',
                'time_steps': batch['time_indices_past'],
                'time_steps_decode':
                    batch['time_indices_past_decode'] if 'time_indices_past_decode' in batch else None,
                'seg_len': batch['video_len_past'],
                'seg_len_decode': batch['video_len_past_decode'] if 'video_len_past_decode' in batch else None,
                'origin': '',
            }), (
            'seg_inp_f', {
                'tensor': batch['future'],
                'space': 'segment',
                'time_label': 'future',
                'time_steps': batch['time_indices_future'],
                'time_steps_decode':
                    batch['time_indices_future_decode'] if 'time_indices_future_decode' in batch else None,
                'seg_len': batch['video_len_future'],
                'seg_len_decode': batch['video_len_future_decode'] if 'video_len_future_decode' in batch else None,
                'origin': '',
            })
        ])

        if return_all:  # all the segment (past and future)
            assert 'all' in batch and not batch['all'].isnan().any()
            dictionary['seg_inp_a'] = {
                'tensor': batch['all'],
                'space': 'segment',
                'time_label': 'all',
                'time_steps': batch['time_indices_all'],
                'time_steps_decode':
                    batch['time_indices_all_decode'] if 'time_indices_all_decode' in batch else None,
                'seg_len': batch['video_len'],
                'seg_len_decode': batch['video_len'] if 'video_len_all_decode' in batch else None,
                'origin': '',
            }

        if 'time_indices_decode' in batch:  # For visualizations
            dictionary['seg_inp_d'] = {
                # May not be associated to any input if we just want to query it
                'tensor': batch['interpolation'] if 'interpolation' in batch else None,
                'space': 'segment',
                'time_label': 'decode',
                'time_steps': batch['time_indices_decode'],
                'seg_len': batch['video_len_decode'],
                'origin': '',
                'splitting': 0,  # No trajectory originates from here
                'to_combine': False  # Do not want to process it, just use it for evaluation or losses
            }

        return cls(dictionary)

    @classmethod
    def create_from_partial(cls, previous_dict: TrajectoryDict, keys: list):
        """
        Create a new dict from a previous one by only taking a few of the elements
        """
        dictionary = OrderedDict([(k, previous_dict[k]) for k in keys])
        return cls(dictionary)

    @classmethod
    def uncombine(cls, video_tensor, video_dict_old, tensor_size, time_steps=None, time_label=None, seg_len=None,
                  last_operation='encode') -> TrajectoryDict:
        """
        Here the video_dict comes from a previous dictionary. We create a new video_dict with the video_tensor, so the
        tensors in video_dict are ignored. We only use its info.
        :param video_tensor: torch.tensor  [B*M, ...]
        :param video_dict_old: dictionary with info about the input to the last operation
        :param tensor_size: size of the input to the last operation
        :param time_steps: if last_operation == 'decode', the time used to decode it. [B*M, Z, T]
        :param time_label: 'past' or 'future' (or None), label of the previous time_steps. [M, Z]
        :param seg_len: actual length of the tensors (without padding). [B*M, Z]
        :param last_operation:
        """
        assert last_operation in ['encode', 'decode']
        video_dict_old = video_dict_old.dictionary
        video_tensor = video_tensor.view(tensor_size[0], -1, *video_tensor.shape[1:])
        time_steps = time_steps.view(tensor_size[0], -1, *time_steps.shape[1:]) if time_steps is not None else None
        seg_len = seg_len.view(tensor_size[0], -1, *seg_len.shape[1:]) if seg_len is not None else None
        assert video_tensor.shape[1] == np.array([v.get('splitting', 1) for v in video_dict_old.values()]).sum()

        video_list = []
        m = 0
        for _, info in video_dict_old.items():
            origin = ''
            if last_operation == 'encode':
                prior_last_operation = 'inp' if info["origin"] == '' else 'dec'
                prior_t = cls.abbr(info['time_label'])
                origin = f'{prior_last_operation}_{prior_t}'
            if info['origin'] != '':
                origin += info["origin"] if origin == '' else f'_{info["origin"]}'
            # Sometimes (e.g. in "intersection", a single trajectory results in multiple segments
            splitting = info.get('splitting', 1)
            for s in range(splitting):
                if last_operation == 'encode':
                    assert time_steps is None and time_label is None and info['space'] == 'segment'
                    new_info = {
                        'tensor': video_tensor[:, m],
                        'space': 'traj',
                        'time_label': None,
                        'time_steps': None,
                        'seg_len': None,
                        'origin': origin
                    }
                    video_list.append(new_info)
                else:  # last_operation == 'decode'
                    assert info['space'] == 'traj'
                    new_info = {
                        'tensor': video_tensor[:, m],
                        'space': 'segment',
                        'time_label': time_label[m],
                        'time_steps': time_steps[:, m],
                        'seg_len': seg_len[:, m],
                        'origin': origin,
                    }
                    video_list.append(new_info)
                m += 1

        dictionary = OrderedDict([(cls.name_tensor(**info), info) for info in video_list])
        return cls(dictionary)

    @classmethod
    def join_dicts(cls, list_dicts) -> TrajectoryDict:
        return cls(OrderedDict(ChainMap(*[d.dictionary for d in list_dicts[::-1]])))

    def combine(self, purpose='encode', reference_dict=None, spatial_crop=False,
                dataset_cls: HumanKeypointsDataset = None, only_combine_one=False, sample_fn=None):
        """
        Given a dictionary, it combines the tensors to be used as input to either encoder or decoder
        :param purpose:
        :param reference_dict: a previous dict from which we may get some info
        :param spatial_crop: bool. If True, the segments that have been decoded are spatially cropped (zero-ed) like the
            original segments they represent. Only used if purpose is 'encode'
        :param dataset_cls: We can use methods in the dataset, to manipulate the 'tensor' attribute
        :param only_combine_one: When we don't really want to decode, just want to combine the representations for other
            purposes. We do not care about the time indices
        :param sample_fn: Samples a trajectory from the distribution of trajectories, so that the returned tensor is
            already sampled. This is convenient if we want to decode different segments (i.e. past and future) from the
            same sampled trajectory, not two samples from the same distributions. Good for visualizations.

        :return:
            videos: [B*M, T, C], where C is the dimensionality of every point in the input. For an actual video, this
                C would be, for example, 3*224*224. For a 2D trajectory, C is equal to 2. B includes both queries and
                targets (forward pass does not discriminate).
            vid_len: B*M
            time_indices: [B*M, T] time indices of each point in the segment (to encode from, or to decode to)
            tensor_size: to store information about the original tensors
        """
        if reference_dict is None:
            reference_dict = self

        videos = []
        videos_len = []
        time_indices = []
        time_labels = []
        tensor_size = None

        for name, element in self.dictionary.items():
            if (element['space'] == ('segment' if purpose == 'encode' else 'traj')) and \
                    (element['tensor'] is not None) and \
                    ('to_combine' not in element or element['to_combine']):

                if sample_fn is not None:
                    assert element['space'] == 'traj'
                    element['tensor'] = sample_fn(element['tensor'])

                if tensor_size is None:
                    tensor_size = element['tensor'].shape[:-1]
                else:
                    assert element['tensor'].shape[:-1] == tensor_size

                if element['time_steps'] is None:
                    # If a tensor in trajectories does not have decoding info, we use the information from the segment
                    # it was encoded from
                    assert purpose == 'decode'
                    time_steps = reference_dict[f'seg_{element["origin"]}']['time_steps']
                    seg_len = reference_dict[f'seg_{element["origin"]}']['seg_len']
                    time_label = reference_dict[f'seg_{element["origin"]}']['time_label']
                else:
                    time_steps = element['time_steps']
                    seg_len = element['seg_len']
                    time_label = element['time_label']

                if only_combine_one and type(time_steps) == list:
                    assert purpose == 'decode', 'Why is time_steps a list?'
                    time_steps, seg_len, time_label = time_steps[0], seg_len[0], time_label[0]

                if type(time_steps) == list:
                    assert purpose == 'decode'  # decode into more than one segment.
                    for i in range(len(time_steps)):
                        time_indices.append(time_steps[i])
                        videos_len.append(seg_len[i])
                        time_labels.append(time_label[i])
                        videos.append(element['tensor'])  # repeat the same trajectory tensor multiple times
                else:
                    time_indices.append(time_steps)
                    videos_len.append(seg_len)
                    time_labels.append(time_label)
                    if spatial_crop and purpose == 'encode':
                        # Take the time that was decoded to
                        name_ref = 'seg_inp_' + name.split('dec_')[1].split('_')[0]
                        assert name_ref in reference_dict
                        reference_tensor = reference_dict[name_ref]['tensor']
                        element_tensor = dataset_cls.spatial_crop(element['tensor'], reference_tensor)
                    else:
                        element_tensor = element['tensor']
                    videos.append(element_tensor)

        # This is not the same as torch.cat, this interleaves the different tensors of every sample in the batch
        videos = torch.stack(videos, dim=1)  # [B, M, ...]
        videos_len = torch.stack(videos_len, dim=1)
        time_indices = torch.stack(time_indices, dim=1)
        videos = videos.view(-1, *videos.shape[2:])  # [B*M, ...]
        videos_len = videos_len.view(-1, *videos_len.shape[2:])
        time_indices = time_indices.view(-1, *time_indices.shape[2:])

        return videos, videos_len, time_indices, time_labels, tensor_size

    @classmethod
    def name_tensor(cls, space, time_label, origin, **kwargs):
        s = 'seg' if space == 'segment' else 'tra'
        name = s
        if space == 'segment':
            last_operation = 'inp' if origin == '' else 'dec'
            t = cls.abbr(time_label)
            name += f'_{last_operation}_{t}'
        if origin != '':
            name += f'_{origin}'
        return name

    @classmethod
    def abbr(cls, x):
        dict_abbr = {'past': 'p', 'future': 'f', 'all': 'a'}
        if x in dict_abbr:
            return dict_abbr[x]
        else:
            return x

    def add_decoding_info(self, key_traj, keys_seg, reference_dict):
        """
        Given a dictionary and a key in that dictionary, it returns a new dictionary where the element from that key
        (key_traj) has been split into several elements, each one with a different decoding information
        The key_traj has to correspond to a trajectory-space element, and if it contains prior decoding information, we
        add to it.
        The decoding information is copied from other keys (in list keys_seg), that correspond to
        segment-space elements.
        reference_dict is the dict where the decoding info comes from

        By default, encoded segments are decoded into the same times they were encoded from. To modify this, use this
        method.
        """
        if reference_dict is None:
            reference_dict = self

        assert key_traj in self.dictionary and self[key_traj]['space'] == 'traj'
        time_label = self[key_traj]['time_label'] if self[key_traj]['time_label'] is not None else []
        time_steps = self[key_traj]['time_steps'] if self[key_traj]['time_steps'] is not None else []
        seg_len = self[key_traj]['seg_len'] if self[key_traj]['seg_len'] is not None else []
        for key_seg in keys_seg:
            assert key_seg in reference_dict and reference_dict[key_seg]['space'] == 'segment'
            time_lab = reference_dict[key_seg]['time_label']
            if time_lab in time_label:
                continue  # We already added this decoding info
            time_label.append(time_lab)
            if ('time_steps_decode' in reference_dict[key_seg]) and \
                    (reference_dict[key_seg]['time_steps_decode'] is not None):
                time_steps_ = reference_dict[key_seg]['time_steps_decode']
            else:
                time_steps_ = reference_dict[key_seg]['time_steps']
            time_steps.append(time_steps_)
            seg_len.append(reference_dict[key_seg]['seg_len'])
        self[key_traj]['time_label'] = time_label
        self[key_traj]['time_steps'] = time_steps
        self[key_traj]['seg_len'] = seg_len
        self[key_traj]['splitting'] = len(time_label)

    def add_trajectory(self, name, tensor, space='traj', time_label=None, time_steps=None, seg_len=None, origin=None):
        self.dictionary[name] = {
            'tensor': tensor,
            'space': space,
            'time_label': time_label,
            'time_steps': time_steps,
            'seg_len': seg_len,
            'origin': origin
        }

    def keys(self):
        return self.dictionary.keys()

    def detach(self):
        for k, v in self.dictionary.items():
            self.dictionary[k]['tensor'] = v['tensor'].detach()

    def __getitem__(self, item):
        return self.dictionary[item]

    def __contains__(self, item):
        return item in self.dictionary
