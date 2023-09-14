# import required typing classes
from typing import List

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go

from stem.geometry import Geometry, Volume, Surface


class PlotUtils:

    @staticmethod
    def __add_2d_surface_to_plot(geometry: 'Geometry',  surface: 'Surface', show_surface_ids: bool,
                                 show_line_ids: bool, show_point_ids: bool,
                                 fig: 'go.Figure') -> npt.NDArray[np.float64]:
        """
        Adds a 2D surface to a plotly graph object figure

        Args:
            - geometry (stem.geometry.Geometry): geometry object
            - surface (stem.geometry.Surface): surface object
            - show_surface_ids (bool): flag to show surface ids
            - show_line_ids (bool): flag to show line ids
            - show_point_ids (bool): flag to show point ids
            - fig (plotly.graph_objects.Figure): graph object figure to which the surface is added

        Returns:
            - NDArray[float]: surface centroid

        """
        # initialize list of surface point ids
        surface_point_ids: List[int] = []

        # calculate centroids of lines to show line ids
        line_centroids = np.zeros((len(surface.line_ids), 3))
        for i, line_k in enumerate(surface.line_ids):

            # get current line
            line = geometry.lines[abs(line_k)]

            # copy line point ids as line_connectivities can be reversed
            line_connectivities = np.copy(line.point_ids)

            # reverse line connectivity if line is defined in opposite direction
            if line_k < 0:
                line_connectivities = line_connectivities[::-1]

            # calculate line centroid
            line_centroids[i, :] = np.mean([geometry.points[line_connectivities[0]].coordinates,
                                               geometry.points[line_connectivities[1]].coordinates], axis=0)

            surface_point_ids.extend(line_connectivities)

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
                                 line={"color": 'black', "width": 1}, marker={"color": 'red', "size": 5},
                                 fill='toself', fillcolor="#ADD8E6"))

        # calculate surface centroid and add to list of all surface centroids which are required to calculate
        # the volume centroid
        surface_centroid: npt.NDArray[np.float64] = np.mean(surface_point_coordinates, axis=0, dtype=np.float64)

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

            fig.add_trace(go.Scatter(x=point_coordinates[:, 0], y=point_coordinates[:,1], mode='text',
                                     text=text_array, textfont={"size": 14}, textposition="top right"))

        return surface_centroid

    @staticmethod
    def __add_3d_surface_to_plot(geometry: 'Geometry', surface: 'Surface', show_surface_ids: bool, show_line_ids: bool,
                                 show_point_ids: bool, fig: 'go.Figure') -> npt.NDArray[np.float64]:
        """
        Adds a 3D surface to a plotly graph object figure.

        Args:
            - geometry (:class:'stem.geometry.Geometry'): geometry object
            - surface (:class:'stem.geometry.Surface'): surface object
            - show_surface_ids (bool): flag to show surface ids
            - show_line_ids (bool): flag to show line ids
            - show_point_ids (bool): flag to show point ids
            - fig (plotly.graph_objects.Figure): graph object figure to which the surface is added

        Returns:
            - NDArray[float]: surface centroid

        """
        # initialize list of surface point ids
        surface_point_ids: List[int] = []

        # calculate centroids of lines to show line ids
        line_centroids = np.zeros((len(surface.line_ids), 3))
        for i, line_k in enumerate(surface.line_ids):

            # get current line
            line = geometry.lines[abs(line_k)]

            # copy line point ids as line_connectivities can be reversed
            line_connectivities = np.copy(line.point_ids)

            # reverse line connectivity if line is defined in opposite direction
            if line_k < 0:
                line_connectivities = line_connectivities[::-1]

            # calculate line centroid
            line_centroids[i, :] = np.mean([geometry.points[line_connectivities[0]].coordinates,
                                               geometry.points[line_connectivities[1]].coordinates], axis=0)

            surface_point_ids.extend(line_connectivities)

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
                                   z=closed_loop_coordinates[:, 2], mode='lines+markers', line={"color": 'black',
                                                                                                "width": 2},
                                   marker={"color": 'red', "size": 2}))

        # calculate surface centroid and add to list of all surface centroids which are required to calculate
        # the volume centroid
        surface_centroid: npt.NDArray[np.float64] = np.mean(surface_point_coordinates, axis=0, dtype=np.float64)

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
    def __add_3d_volume_to_plot(geometry: 'Geometry', volume: 'Volume', show_volume_ids: bool, show_surface_ids: bool,
                                show_line_ids: bool, show_point_ids: bool, fig: 'go.Figure'):
        """
        Adds a 3D volume to a plotly graph object figure.

        Args:
            - geometry (:class:`stem.geometry.Geometry`): Geometry object
            - volume (:class:`stem.geometry.Volume`): Volume object
            - show_volume_ids (bool): Show volume ids
            - show_surface_ids (bool): Show surface ids
            - show_line_ids (bool): Show line ids
            - show_point_ids (bool): Show point ids
            - fig (plotly.graph_objects.Figure): graph object figure to which the surface is added

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
    def create_geometry_figure(ndim: int, geometry: 'Geometry', show_volume_ids: bool = False,
                               show_surface_ids: bool = False, show_line_ids: bool = False,
                               show_point_ids: bool = False) -> 'go.Figure':
        """
        Creates the geometry of the model in a plotly graph object figure.

        Args:
            - ndim (int): Number of dimensions of the geometry. Either 2 or 3.
            - geometry (:class:`stem.geometry.Geometry`): Geometry object.
            - show_volume_ids (bool): If True, the volume ids are shown in the plot.
            - show_surface_ids (bool): If True, the surface ids are shown in the plot.
            - show_line_ids (bool): If True, the line ids are shown in the plot.
            - show_point_ids (bool): If True, the point ids are shown in the plot.

        Returns:
            - plotly.graph_objects.Figure: graph object figure

        """

        # Initialize figure
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

        # set limits of plot
        # extend limits with buffer, which is 10% of the difference between min and max
        buffer = 0.1
        all_coordinates = np.array([point.coordinates for point in geometry.points.values()])

        # calculate and set x and y limits
        min_x, max_x = np.min(all_coordinates[:, 0]), np.max(all_coordinates[:, 0])
        dx = max_x - min_x
        min_y, max_y = np.min(all_coordinates[:, 1]), np.max(all_coordinates[:, 1])
        dy = max_y - min_y

        xlim = [min_x - buffer * dx, max_x + buffer * dx]
        ylim = [min_y - buffer * dy, max_y + buffer * dy]

        # set scene for 2D or 3D
        if ndim == 2:

            fig.update_layout(xaxis={"title": "x-coordinates [m]",
                                     "range": xlim},
                              yaxis={"title": "y-coordinates [m]",
                                     "range": ylim},
                              showlegend=False,
                              hovermode=False)

        elif ndim == 3:

            # calculate and set z limits
            min_z, max_z = np.min(all_coordinates[:, 2]), np.max(all_coordinates[:, 2])
            dz = max_z - min_z

            zlim = [min_z - buffer * dz, max_z + buffer * dz]

            scene = dict(xaxis={"title": "x-coordinates [m]",
                                "range": xlim},
                         yaxis={"title": "y-coordinates [m]",
                                "range": ylim},
                         zaxis={"title": "z-coordinates [m]",
                                "range": zlim},
                         camera={"up": {"x": 0, "y": 1, "z": 0}})
            fig.update_layout(scene=scene)

        else:
            raise ValueError("Number of dimensions should be 2 or 3")

        # set layout
        fig.update_layout(showlegend=False,
                          hovermode=False)

        return fig
