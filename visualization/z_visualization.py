import numpy as np
import functools
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from z_curve_computation.z_curve_magic import variables_magic_numbers, generate_magic_number_arrays
from utils.helping_methods import classic_nd_variables
# from z_curve_computation.z_curve_array_manipulation import variables_array_manipulation


def curves_comparison_2d():
    coordinate_np_type = np.uint16
    index_np_type = np.uint32
    var_count = 2
    grid_modulo = 16
    visualisation_args = {"z_index_colormap": 'jet', "labels_font_size": 10, "labels_modulo_val": 10, "z_index_points_size": 3, "legend_bbox_to_anchor": (-0.18, 1.08)}

    curves_comparison_general(coordinate_np_type, index_np_type, var_count, grid_modulo, visualisation_args.values())


def curves_comparison_3d():
    coordinate_np_type = np.uint16
    index_np_type = np.uint64
    var_count = 3
    grid_modulo = 4
    visualisation_args = {'z_index_colormap': 'jet', 'labels_font_size': 11, 'labels_modulo_val': 10, 'z_index_points_size': 12, 'legend_bbox_to_anchor': (-0.19, 1)}

    curves_comparison_general(coordinate_np_type, index_np_type, var_count, grid_modulo, visualisation_args.values())


def distances_comparison():
    coordinate_np_type = np.uint8
    index_np_type = np.uint64
    var_count = 8
    modulo_grid = 6
    # var_func = functools.partial(variables_array_manipulation, coord_type=coordinate_np_type, index_type=index_np_type,
    #                             var_count=var_count)
    B, S = generate_magic_number_arrays(coordinate_np_type, index_np_type, var_count)

    var_func = functools.partial(variables_magic_numbers, coord_type=coordinate_np_type, index_type=index_np_type,
                                 var_count=var_count, B=B, S=S)
    classic_var_func = functools.partial(classic_nd_variables, mod_val=modulo_grid, n_dim=var_count)
    distances_visualisation(var_func, True, [0, modulo_grid**var_count], index_np_type, var_count, modulo_grid, "Succesor distances: Z-index")
    distances_visualisation(classic_var_func, True, [0, modulo_grid**var_count], index_np_type, var_count, modulo_grid, "Succesor distances:  standard index")
    plt.show()


def curves_comparison_general(coordinate_np_type, index_np_type, var_count, grid_modulo, visualisation_args):
    B, S = generate_magic_number_arrays(coordinate_np_type, index_np_type, var_count)
    var_func = functools.partial(variables_magic_numbers, coord_type=coordinate_np_type, index_type=index_np_type,
                                 var_count=var_count, B=B, S=S)
    # var_func = functools.partial(variables_array_manipulation, coord_type=coordinate_np_type,
    #                             index_type=index_np_type, var_count=var_count)

    classic_var_func = functools.partial(classic_nd_variables, mod_val=grid_modulo, n_dim=var_count)
    z_index_visualisation(var_func, True, [0, grid_modulo ** var_count], coordinate_np_type, index_np_type,
                          str(var_count)+"D z-order curve", *visualisation_args)
    z_index_visualisation(classic_var_func, True, [0, grid_modulo ** var_count], coordinate_np_type, index_np_type,
                          str(var_count)+"D standard curve", *visualisation_args)
    plt.show()


def z_index_visualisation(var_function, vector_function, bounds,coord_np_type, index_np_type, title, z_index_colormap='jet', labels_font_size=11, labels_modulo_val=10, z_index_points_size=12, legend_bbox_to_anchor=(-0.19, 1)):
    if vector_function:
        vars_variables = np.array(var_function(np.array(range(bounds[0], bounds[1]), dtype = coord_np_type)))
    else:
        vars_variables = np.array([var_function(index_np_type(z)) for z in range(bounds[0], bounds[1])], dtype=coord_np_type)

    data_length = len(vars_variables)

    if z_index_colormap is None or len(z_index_colormap) == 0:
        z_index_colors = ['#1f77b4' for _ in range(data_length)]
    else:
        z_index_colors = plt.cm.get_cmap(z_index_colormap)(np.linspace(0,  1, data_length))

    var_count = len(vars_variables[0])
    projection_str = '3d' if var_count == 3 else None
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=projection_str)
    axis_labels = ["x","y","z"]
    for i in range(var_count):
        axis = getattr(ax,axis_labels[i]+"axis")
        axis.set_major_locator(MaxNLocator(integer=True))
        axis.set_label("Variable "+str(i))

    ax.scatter(*vars_variables.T, c=z_index_colors, label='Points with z'+r"$\in$"+"{"+r"${x},\dots,{y}$".format(x=str(bounds[0]), y=str(bounds[1]-1))+"}", s=z_index_points_size)
    ax.set_title(title)

    # visualizes text labels only for indexes which are divisible by modulo_val
    data_strings = [(i*labels_modulo_val, str(i*labels_modulo_val)) for i in range(data_length//labels_modulo_val+1)]
    for index, txt in data_strings:
        ax.text(*vars_variables[index], txt, fontsize=labels_font_size)
    ax.legend(loc='upper left', bbox_to_anchor=legend_bbox_to_anchor, framealpha=1)
    for i in range(data_length-1):
        plt.plot(*vars_variables[i:i+2].T, c=z_index_colors[i])

def distances_visualisation(var_function, vector_function, bounds, index_func, var_count, grid_width, title):
    counting_type = np.int64
    if vector_function:
        vars_variables = np.array(var_function(np.array(range(bounds[0], bounds[1]), dtype=index_func)), dtype=counting_type)
    else:
        vars_variables = np.array([var_function(index_func(z)) for z in range(bounds[0], bounds[1])], dtype=counting_type)

    differences = np.diff(vars_variables, axis=0)
    distances = np.linalg.norm(differences, axis=1)
    fig, ax = plt.subplots()
    ax.hist(distances, label="Total distance = "+str(int(np.sum(distances)))+"\n"+"Variables count: "+str(var_count)+", grid "+str(grid_width))
    ax.set_yscale('log')
    ax.set_title(title)
    ax.set_xlabel("Distance to succesor")
    ax.set_ylabel("Count")
    ax.legend()

