from nilearn import datasets, plotting, surface
import matplotlib.pyplot as plt
import numpy as np
import pkg_resources
from matplotlib.colors import LinearSegmentedColormap


def get_dk_surface_data(data_vector, hemi, resolution='fsaverage5', exclude=[0, 4]):
    '''
    The data vector is e.g. 72 regions  in freesurfer order (1000-2035):
    https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT
    If you have e.g. 68 regions in the data vector because corpus callosum and unknown 
    are not included then exclude =[0,4]
    resolution can either be standard (fsaverage5) or highres (fsaverage)
    '''
    fs_average = datasets.fetch_surf_fsaverage(mesh=resolution)
    labels = {'left': surface.load_surf_data(pkg_resources.resource_filename(__name__, f'data/{resolution}/lh.aparc.annot')),
              'right': surface.load_surf_data(pkg_resources.resource_filename(__name__, f'data/{resolution}/rh.aparc.annot'))}
    
    comp_labels = np.zeros(labels[hemi].shape[0])
    remove_idx = np.in1d(np.arange(36), exclude)
    indices = np.arange(36)[~remove_idx]

    for i in exclude:
        idx = labels[hemi] == i
        comp_labels[idx] = np.nan

    for i, j in enumerate(indices):
        k=i
        if hemi == "right":
             k = i + 36-len(exclude)
        idx = labels[hemi] == j
        comp_labels[idx] = data_vector[k]

    surface_data = {'comp_labels': comp_labels,
                    'surf_mesh': fs_average[f'infl_{hemi}'],
                    'bg_maps': fs_average[f'sulc_{hemi}']}

    return surface_data


def plot_surf(surface_data, view, axes, fig, vmax=1, vmin=-1):
    # cmap = LinearSegmentedColormap.from_list('custom blue', 
    #                                         [(0,   '#2166ac' ),
    #                                         (0.5, '#ffffff'),
    #                                         (1,    '#b2182b')], N=256)
    cmap = 'RdBu'
    img = plotting.plot_surf(surface_data['surf_mesh'], 
                        surf_map=surface_data['comp_labels'],
                        hemi=view[0], view=view[1],
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        bg_map=surface_data['bg_maps'],
                        darkness=0.6,
                        bg_on_data=True,
                        title='',
                        figure = fig,
                        axes = axes                        
                            )


def plot_single(view, data_vector, axes, fig, resolution='fsaverage5', vmax=1, vmin=-1):
    surface_data = get_dk_surface_data(data_vector, view[0])
    plot_surf(surface_data, view, axes, fig, vmax=vmax, vmin=vmin)


def plot_grid(data_vector, vmax=1, vmin=-1):
    plt.rcParams['figure.figsize'] = [8, 8]
    fig, axs = plt.subplots(nrows=2, ncols=2,
                            subplot_kw={'projection': '3d'},
                            gridspec_kw={'wspace': 0, 'hspace': 0})
    view = ('left', 'lateral')
    plot_single(view, data_vector, axs[0][0], fig,vmax=vmax, vmin=vmin)
    view = ('left', 'medial')
    plot_single(view, data_vector, axs[1][0], fig,vmax=vmax, vmin=vmin)
    view = ('right', 'lateral')
    plot_single(view, data_vector, axs[0][1], fig,vmax=vmax, vmin=vmin)
    view = ('right', 'medial')
    plot_single(view, data_vector, axs[1][1], fig,vmax=vmax, vmin=vmin)
    plt.tight_layout()
