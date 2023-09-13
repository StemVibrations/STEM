import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Axes3D import has side effects, it enables using projection='3d' in add_subplot
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, PolyCollection
import plotly.graph_objects as go

# import required typing classes
from typing import TYPE_CHECKING, List, Optional

from stem.mesh import Mesh
from stem.model_part import BodyModelPart, ModelPart

if TYPE_CHECKING:
    from stem.geometry import Geometry, Volume, Surface


class PlotUtils:

    @staticmethod
    def __add_2d_surface_to_plot(geometry: 'Geometry',  surface: 'Surface', show_surface_ids,
                                 show_line_ids, show_point_ids, fig):
        """
        Adds a 2D surface to a plot

        Args:
            - geometry (stem.geometry.Geometry): geometry object
            - surface (stem.geometry.Surface): surface object
            - show_surface_ids (bool): flag to show surface ids
            - show_line_ids (bool): flag to show line ids
            - show_point_ids (bool): flag to show point ids
            - ax (matplotlib.axes.Axes): axes object to which the surface is added

        Returns:
            - NDArray[float]: surface centroid

        """
        # initialize list of surface point ids
        surface_point_ids: List[int] = []

        # calculate centroids of lines to show line ids
        line_centroids = []
        for line_k in surface.line_ids:

            # get current line
            line = geometry.lines[abs(line_k)]

            # copy line point ids as line_connectivities can be reversed
            line_connectivities = np.copy(line.point_ids)

            # reverse line connectivity if line is defined in opposite direction
            if line_k < 0:
                line_connectivities = line_connectivities[::-1]

            # calculate line centroid
            line_centroids.append(np.mean([geometry.points[line_connectivities[0]].coordinates,
                                           geometry.points[line_connectivities[1]].coordinates], axis=0))

            surface_point_ids.extend(line_connectivities)

        line_centroids = np.array(line_centroids)

        # get unique points within surface
        unique_points = []
        for point_id in surface_point_ids:

            if point_id not in unique_points:
                unique_points.append(point_id)

        # get coordinates of surface points
        surface_point_coordinates = np.array([geometry.points[point_id].coordinates
                                              for point_id in unique_points])

        # add surface to plot
        closed_loop_coordinates = np.vstack((surface_point_coordinates, surface_point_coordinates[0, :]))
        fig.add_trace(go.Scatter(x=closed_loop_coordinates[:, 0], y=closed_loop_coordinates[:, 1], mode='lines+markers',
                                 line={"color": 'black', "width": 1}, marker= {"color": 'red', "size": 5},
                                 fill='toself', fillcolor="#ADD8E6"))

        # calculate surface centroid and add to list of all surface centroids which are required to calculate
        # the volume centroid
        surface_centroid = np.mean(surface_point_coordinates, axis=0)

        # show surface ids
        if show_surface_ids:
            fig.add_trace(go.Scatter(x=[surface_centroid[0]], y=[surface_centroid[1]], mode='text',
                                     text=f"<b>s_{abs(surface.id)}</b>", textfont={"size": 14},
                                     textposition="middle center"))

        # show line ids
        if show_line_ids:

            text_array = [f"<b>l_{abs(line_k)}</b>" for line_k in surface.line_ids]

            fig.add_trace(go.Scatter(x=line_centroids[:, 0], y=line_centroids[:, 1], mode='text',
                                     text=text_array, textfont={"size": 14},
                                     textposition="top right"))

        # show point ids
        if show_point_ids:

            text_array = [f"<b>p_{point_id}</b>" for point_id in geometry.points.keys()]
            point_coordinates = np.array([point.coordinates for point in geometry.points.values()])

            fig.add_trace(go.Scatter(x=point_coordinates[:,0], y=point_coordinates[:,1], mode='text',
                                     text=text_array, textfont={"size": 14}, textposition="top right"))

        return surface_centroid


    @staticmethod
    def __add_3d_surface_to_plot(geometry: 'Geometry', surface: 'Surface', show_surface_ids, show_line_ids,
                                 show_point_ids, fig):
        """
        Adds a 3D surface to a plot.

        Args:
            - geometry (stem.geometry.Geometry): geometry object
            - surface (stem.geometry.Surface): surface object
            - show_surface_ids (bool): flag to show surface ids
            - show_line_ids (bool): flag to show line ids
            - show_point_ids (bool): flag to show point ids
            - ax (mpl_toolkits.mplot3d.axes3d.Axes3D): axes object to which the surface is added

        Returns:
            - NDArray[float]: surface centroid

        """
        # initialize list of surface point ids
        surface_point_ids: List[int] = []

        # calculate centroids of lines to show line ids
        line_centroids = []
        for line_k in surface.line_ids:

            # get current line
            line = geometry.lines[abs(line_k)]

            # copy line point ids as line_connectivities can be reversed
            line_connectivities = np.copy(line.point_ids)

            # reverse line connectivity if line is defined in opposite direction
            if line_k < 0:
                line_connectivities = line_connectivities[::-1]

            # calculate line centroid
            line_centroids.append(np.mean([geometry.points[line_connectivities[0]].coordinates,
                                           geometry.points[line_connectivities[1]].coordinates], axis=0))

            surface_point_ids.extend(line_connectivities)

        line_centroids = np.array(line_centroids)

        # get unique points within surface
        unique_points = []
        for point_id in surface_point_ids:

            if point_id not in unique_points:
                unique_points.append(point_id)

        # get coordinates of surface points
        surface_point_coordinates = np.array([geometry.points[point_id].coordinates
                                              for point_id in unique_points])

        # check which delaunay axis to use for the meshing
        delaunayaxis = 'z'
        if np.allclose(surface_point_coordinates[:, 0], surface_point_coordinates[0, 0]):
            delaunayaxis = 'x'
        elif np.allclose(surface_point_coordinates[:, 1], surface_point_coordinates[0, 1]):
            delaunayaxis = 'y'

        # add surface to plot
        fig.add_trace(go.Mesh3d(x=surface_point_coordinates[:, 0], y=surface_point_coordinates[:, 1],
                                z=surface_point_coordinates[:, 2], opacity=0.25, showscale=False,
                                delaunayaxis=delaunayaxis, color='blue'))

        # add surface edges to plot
        closed_loop_coordinates = np.vstack((surface_point_coordinates, surface_point_coordinates[0, :]))
        fig.add_trace(go.Scatter3d(x=closed_loop_coordinates[:, 0], y=closed_loop_coordinates[:, 1],
                                   z=closed_loop_coordinates[:, 2], mode='lines+markers', line={"color": 'black', "width": 2},
                                   marker={"color": 'red', "size": 2}))

        # calculate surface centroid and add to list of all surface centroids which are required to calculate
        # the volume centroid
        surface_centroid = np.mean(surface_point_coordinates, axis=0)

        # show surface ids
        if show_surface_ids:
            fig.add_trace(go.Scatter3d(x=[surface_centroid[0]], y=[surface_centroid[1]],
                                       z=[surface_centroid[2]], mode='text',
                                       text=f"<b>s_{abs(surface.id)}</b>", textfont={"size": 14},
                                       textposition="middle center"))

        # show line ids
        if show_line_ids:

            text_array = [f"<b>l_{abs(line_k)}</b>" for line_k in surface.line_ids]

            fig.add_trace(go.Scatter3d(x=line_centroids[:,0], y=line_centroids[:,1], z=line_centroids[:,2], mode='text',
                                       text=text_array, textfont={"size": 14},
                                       textposition="middle center"))

        # show point ids
        if show_point_ids:

            text_array = [f"<b>p_{point_id}</b>" for point_id in geometry.points.keys()]
            point_coordinates = np.array([point.coordinates for point in geometry.points.values()])

            fig.add_trace(go.Scatter3d(x=point_coordinates[:,0], y=point_coordinates[:,1],
                                       z=point_coordinates[:,2], mode='text',
                                       text=text_array, textfont={"size": 14},
                                       textposition="middle center"))

        # return data, surface
        return surface_centroid

    @staticmethod
    def __add_3d_volume_to_plot(geometry: 'Geometry', volume: 'Volume', show_volume_ids, show_surface_ids,
                                show_line_ids, show_point_ids, fig):
        """
        Adds a 3D volume to a matplotlib figure.

        Args:
            - geometry (:class:`stem.geometry.Geometry`): Geometry object
            - volume (:class:`stem.geometry.Volume`): Volume object
            - show_volume_ids (bool): Show volume ids
            - show_surface_ids (bool): Show surface ids
            - show_line_ids (bool): Show line ids
            - show_point_ids (bool): Show point ids
            - ax (mpl_toolkits.mplot3d.axes3d.Axes3D): Axes object to which the volume is added

        """
        # initialize list of surface centroids which are required to plot the surface ids
        all_surface_centroids = []

        # loop over all surfaces within the volume
        for surface_k in volume.surface_ids:
            # get current surface
            surface = geometry.surfaces[abs(surface_k)]

            surface_centroid = PlotUtils.__add_3d_surface_to_plot(geometry, surface, show_surface_ids, show_line_ids,
                                                             show_point_ids, fig)
            all_surface_centroids.append(surface_centroid)
        # show volume ids
        if show_volume_ids:
            volume_centroid = np.mean(all_surface_centroids, axis=0)
            fig.add_trace(go.Scatter3d(x=[volume_centroid[0]], y=[volume_centroid[1]], z=[volume_centroid[2]],
                                       mode='text', text=f"<b>v_{volume.id}</b>", textfont={"size": 18},
                                       textposition="middle center"))

    @staticmethod
    def create_geometry_figure(ndim: int, geometry: 'Geometry', show_volume_ids: bool = False, show_surface_ids: bool = False,
                               show_line_ids: bool = False, show_point_ids: bool = False) -> go.Figure:
        """
        Creates the geometry of the model in a matplotlib plot.

        Args:
            - ndim (int): Number of dimensions of the geometry. Either 2 or 3.
            - geometry (:class:`stem.geometry.Geometry`): Geometry object.
            - show_volume_ids (bool): If True, the volume ids are shown in the plot.
            - show_surface_ids (bool): If True, the surface ids are shown in the plot.
            - show_line_ids (bool): If True, the line ids are shown in the plot.
            - show_point_ids (bool): If True, the point ids are shown in the plot.

        Returns:
            - plt.Figure: Figure object

        """



        # Initialize figure in 3D
        # fig = plt.figure()
        fig = go.Figure()

        if ndim == 2:
            for surface in geometry.surfaces.values():

                PlotUtils.__add_2d_surface_to_plot(geometry, surface, show_surface_ids, show_line_ids,
                                                   show_point_ids, fig)

        elif ndim == 3:
            # loop over all volumes
            for volume_data in geometry.volumes.values():

                PlotUtils.__add_3d_volume_to_plot(geometry, volume_data, show_volume_ids, show_surface_ids,
                                                  show_line_ids, show_point_ids, fig)

        else:
            raise ValueError("Number of dimensions should be 2 or 3")

        fig.update_layout(scene=dict(xaxis={"title": "x-coordinates [m]"},
                                     yaxis={"title": "y-coordinates [m]"},
                                     zaxis={"title": "z-coordinates [m]"}),
                          showlegend=False)

        fig.write_html("test2.html")

        return fig

if __name__ == '__main__':

    pass
    # import plotly.graph_objects as go
    #
    # # Create data for your 3D mesh plot
    # # Example data
    # x = [0, 1, 2, 3]
    # y = [0, 1, 2, 3]
    # z = [
    #     [1, 2, 1, 2],
    #     [2, 3, 2, 3],
    #     [3, 4, 3, 4],
    #     [4, 5, 4, 5]
    # ]
    #
    # # Create the 3D mesh plot
    # fig = go.Figure()
    #
    # # Add the 3D mesh plot
    # fig.add_trace(go.Mesh3d(x=x, y=y, z=z, opacity=0.7))
    #
    # # # Add planes on the x-y, x-z, and y-z planes
    # # fig.add_trace(go.Surface(x=x, y=y, z=[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], colorscale='viridis',
    # #                          showscale=False))
    # # fig.add_trace(go.Surface(x=x, y=[0, 0, 0, 0], z=z, colorscale='viridis', showscale=False))
    # # fig.add_trace(go.Surface(x=[0, 0, 0, 0], y=y, z=z, colorscale='viridis', showscale=False))
    #
    # # Customize the layout if needed
    # fig.update_layout(
    #     scene=dict(
    #         xaxis=dict(nticks=4),
    #         yaxis=dict(nticks=4),
    #         zaxis=dict(nticks=4),
    #     )
    # )
    #
    # # Show the figure
    # fig.show()