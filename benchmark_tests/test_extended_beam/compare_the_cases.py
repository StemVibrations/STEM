import json
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import itertools


def plot_data(node_id, normal_node_id, coordinates, results_normal, results_extended, data_normal, data_extended,
              data_type, file_suffix):
    """
    General plotting function for displacement or velocity data.

    Parameters:
    - node_id: ID of the current node being plotted.
    - normal_node_id: ID of the closest normal node.
    - coordinates: Coordinates of the node.
    - results_normal: Dictionary of normal case results (contains time data).
    - results_extended: Dictionary of extended case results (contains time data).
    - data_normal: Dictionary containing normal case data (displacements or velocities).
    - data_extended: Dictionary containing extended case data for the node (displacements or velocities).
    - data_type: Type of data being plotted ('DISPLACEMENT' or 'VELOCITY').
    - file_suffix: Suffix for the output file (e.g., 'displacements' or 'velocities').
    """
    fig, axs = plt.subplots(1, 1, figsize=(12, 6), sharex=True)
    # Plot Y component
    axs.plot(results_normal.get("TIME"), data_normal.get(f"{data_type}_Y"), label="Normal")
    axs.plot(results_extended.get("TIME"), data_extended.get(f"{data_type}_Y"), label="Extended")
    axs.set_title(f"{data_type.capitalize()} Y")

    # Add legends, labels, and title
    axs.legend()
    axs.set_xlabel("Time [s]")

    # Add the main title
    plt.suptitle(
        f"Nodes {node_id} of the extended beam test case \n (closest to node {normal_node_id}) \n coordinates: {coordinates} [m]"
    )

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f"compare_{node_id}_{file_suffix}.png")
    plt.close()


if __name__ == "__main__":
    # results for the extended beam test case
    expected_output_dir_temp = r"inputs_kratos_extended\json_output.json"
    nodes_json = r"inputs_kratos_extended/soil_2_nodes.json"
    # read the json file
    with open(expected_output_dir_temp, "r") as f:
        data = json.load(f)
    # read the nodes json file
    with open(nodes_json, "r") as f:
        data_nodes = json.load(f)
    # find the coordinates of the nodes mentioned in the data
    coordinates = {node: data_nodes.get(node) for node in data.keys() if "TIME" not in node}
    # results for the normal case
    expected_output_dir_temp = r"inputs_kratos_full\json_output.json"
    nodes_json = r"inputs_kratos_full/soil_2_nodes.json"
    # read the json file
    with open(expected_output_dir_temp, "r") as f:
        data_full_geometry = json.load(f)
    # read the nodes json file
    with open(nodes_json, "r") as f:
        data_nodes_full = json.load(f)
    # find the coordinates of the nodes mentioned in the data
    coordinates_full = {node: data_nodes_full.get(node) for node in data_full_geometry.keys() if "TIME" not in node}
    for node, coord in coordinates.items():
        # get the equivalent node from the full geometry
        node_full = [key for key, value in coordinates_full.items() if value == coord][0]
        # now plot the displacements
        plot_data(node, node_full, coord, data_full_geometry, data, data_full_geometry[node_full], data[node],
                  "DISPLACEMENT", "displacements")
        plot_data(node, node_full, coord, data_full_geometry, data, data_full_geometry[node_full], data[node],
                  "VELOCITY", "velocity")
