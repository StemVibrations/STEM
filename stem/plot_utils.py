import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go

from stem.geometry import Geometry, Volume, Surface


class PlotUtils:
    """
    Utility class for plotting geometry.
    """

    @staticmethod
    def __add_2d_lines_to_plot(geometry: 'Geometry', show_line_ids: bool, show_point_ids: bool, fig: 'go.Figure'):
        """
        Adds lines to a 2D plotly graph object figure.

        Args:
            - geometry (:class:`stem.geometry.Geometry`): Geometry object.
            - show_line_ids (bool): If True, the line ids are shown in the plot.
            - show_point_ids (bool): If True, the point ids are shown in the plot.
            - fig (plotly.graph_objects.Figure): graph object figure to which the lines are added.

        """

        text_array = []
        line_centroids = []

        for line_id, line in geometry.lines.items():
            point1 = geometry.points[line.point_ids[0]].coordinates
            point2 = geometry.points[line.point_ids[1]].coordinates

            # yapf: disable
            fig.add_trace(
                go.Scatter(x=[point1[0], point2[0]],
                           y=[point1[1], point2[1]],
                           mode='lines+markers',
                           line={"color": 'black', "width": 1},
                           marker={"color": 'red', "size": 5}))
            # yapf: enable

            # show line ids
            if show_line_ids:

                line_centroids.append(geometry.calculate_centroid_of_line(line_id))
                text_array.append(f"<b>l_{abs(line_id)}</b>")

        if show_line_ids:
            fig.add_trace(
                go.Scatter(x=np.array(line_centroids)[:, 0],
                           y=np.array(line_centroids)[:, 1],
                           mode='text',
                           text=text_array,
                           textfont={"size": 14},
                           textposition="top right"))

        # show point ids
        if show_point_ids:

            text_array = [f"<b>p_{point_id}</b>" for point_id in geometry.points.keys()]
            point_coordinates = np.array([point.coordinates for point in geometry.points.values()])

            fig.add_trace(
                go.Scatter(x=point_coordinates[:, 0],
                           y=point_coordinates[:, 1],
                           mode='text',
                           text=text_array,
                           textfont={"size": 14},
                           textposition="top right"))

    @staticmethod
    def __add_2d_surface_to_plot(geometry: 'Geometry', surface: 'Surface', show_surface_ids: bool, fig: 'go.Figure'):
        """
        Adds a 2D surface to a plotly graph object figure

        Args:
            - geometry (:class:`stem.geometry.Geometry`): geometry object
            - surface (:class:`stem.geometry.Surface`): surface object
            - show_surface_ids (bool): flag to show surface ids
            - fig (plotly.graph_objects.Figure): graph object figure to which the surface is added

        """
        surface_points = geometry.get_ordered_points_from_surface(surface.id)
        surface_point_coordinates = np.array([point.coordinates for point in surface_points])

        # add surface to plot
        closed_loop_coordinates = np.vstack((surface_point_coordinates, surface_point_coordinates[0, :]))

        # yapf: disable
        fig.add_trace(go.Scatter(x=closed_loop_coordinates[:, 0], y=closed_loop_coordinates[:, 1], mode='lines+markers',
                                 line={"color": 'black', "width": 1}, marker={"color": 'red', "size": 5},
                                 fill='toself', fillcolor="#ADD8E6"))
        # yapf: enable

        # show surface ids
        if show_surface_ids:
            # calculate surface centre of mass  and add surface id in centre
            surface_centre: npt.NDArray[np.float64] = geometry.calculate_centre_of_mass_surface(surface.id)

            fig.add_trace(
                go.Scatter(x=[surface_centre[0]],
                           y=[surface_centre[1]],
                           mode='text',
                           text=f"<b>s_{abs(surface.id)}</b>",
                           textfont={"size": 14},
                           textposition="middle center"))

    @staticmethod
    def __add_3d_lines_to_plot(geometry: 'Geometry', show_line_ids: bool, show_point_ids: bool, fig: 'go.Figure'):
        """
        Adds lines to a 3D plotly graph object figure.

        Args:
            - geometry (:class:`stem.geometry.Geometry`): Geometry object.
            - show_line_ids (bool): If True, the line ids are shown in the plot.
            - show_point_ids (bool): If True, the point ids are shown in the plot.
            - fig (plotly.graph_objects.Figure): graph object figure to which the lines are added.

        """

        text_array = []
        line_centroids = []

        for line_id, line in geometry.lines.items():
            point1 = geometry.points[line.point_ids[0]].coordinates
            point2 = geometry.points[line.point_ids[1]].coordinates

            # yapf: disable
            fig.add_trace(
                go.Scatter3d(x=[point1[0], point2[0]],
                             y=[point1[1], point2[1]],
                             z=[point1[2], point2[2]],
                             mode='lines+markers',
                             line={"color": 'black', "width": 1},
                             marker={"color": 'red', "size": 5}))
            # yapf: enable

            # show line ids
            if show_line_ids:

                line_centroids.append(geometry.calculate_centroid_of_line(line_id))
                text_array.append(f"<b>l_{abs(line_id)}</b>")

        if show_line_ids:
            fig.add_trace(
                go.Scatter3d(x=np.array(line_centroids)[:, 0],
                             y=np.array(line_centroids)[:, 1],
                             z=np.array(line_centroids)[:, 2],
                             mode='text',
                             text=text_array,
                             textfont={"size": 14},
                             textposition="top center"))

        # show point ids
        if show_point_ids:

            text_array = [f"<b>p_{point_id}</b>" for point_id in geometry.points.keys()]
            point_coordinates = np.array([point.coordinates for point in geometry.points.values()])

            fig.add_trace(
                go.Scatter3d(x=point_coordinates[:, 0],
                             y=point_coordinates[:, 1],
                             z=point_coordinates[:, 2],
                             mode='text',
                             text=text_array,
                             textfont={"size": 14},
                             textposition="top right"))

    @staticmethod
    def __add_3d_surface_to_plot(geometry: 'Geometry', surface: 'Surface', show_surface_ids: bool, fig: 'go.Figure'):
        """
        Adds a 3D surface to a plotly graph object figure.

        Args:
            - geometry (:class:'stem.geometry.Geometry'): geometry object
            - surface (:class:'stem.geometry.Surface'): surface object
            - show_surface_ids (bool): flag to show surface ids
            - fig (plotly.graph_objects.Figure): graph object figure to which the surface is added

        """

        surface_points = geometry.get_ordered_points_from_surface(surface.id)
        surface_point_coordinates = np.array([point.coordinates for point in surface_points])

        # check which delaunay axis to use for the meshing
        delaunayaxis = 'z'
        if np.allclose(surface_point_coordinates[:, 0], surface_point_coordinates[0, 0]):
            delaunayaxis = 'x'
        elif np.allclose(surface_point_coordinates[:, 1], surface_point_coordinates[0, 1]):
            delaunayaxis = 'y'

        # add surface to plot
        fig.add_trace(
            go.Mesh3d(x=surface_point_coordinates[:, 0],
                      y=surface_point_coordinates[:, 1],
                      z=surface_point_coordinates[:, 2],
                      opacity=0.25,
                      showscale=False,
                      delaunayaxis=delaunayaxis,
                      color='blue'))

        # show surface ids
        if show_surface_ids:
            # calculate surface centre of mass
            surface_centre: npt.NDArray[np.float64] = geometry.calculate_centre_of_mass_surface(surface.id)

            fig.add_trace(
                go.Scatter3d(x=[surface_centre[0]],
                             y=[surface_centre[1]],
                             z=[surface_centre[2]],
                             mode='text',
                             text=f"<b>s_{abs(surface.id)}</b>",
                             textfont={"size": 14},
                             textposition="middle center"))

    @staticmethod
    def __add_3d_volume_to_plot(geometry: 'Geometry', volume: 'Volume', show_volume_ids: bool, show_surface_ids: bool,
                                fig: 'go.Figure'):
        """
        Adds a 3D volume to a plotly graph object figure.

        Args:
            - geometry (:class:`stem.geometry.Geometry`): Geometry object
            - volume (:class:`stem.geometry.Volume`): Volume object
            - show_volume_ids (bool): Show volume ids
            - show_surface_ids (bool): Show surface ids
            - fig (plotly.graph_objects.Figure): graph object figure to which the surface is added

        """

        # loop over all surfaces within the volume
        for surface_k in volume.surface_ids:
            # get current surface
            surface = geometry.surfaces[abs(surface_k)]

            PlotUtils.__add_3d_surface_to_plot(geometry, surface, show_surface_ids, fig)

        # show volume ids
        if show_volume_ids:
            volume_centre = geometry.calculate_centre_of_mass_volume(volume.id)
            fig.add_trace(
                go.Scatter3d(x=[volume_centre[0]],
                             y=[volume_centre[1]],
                             z=[volume_centre[2]],
                             mode='text',
                             text=f"<b>v_{volume.id}</b>",
                             textfont={"size": 18},
                             textposition="middle center"))

    @staticmethod
    def __move_labels_to_front(fig: 'go.Figure'):
        """
        Moves the text labels in a plotly graph object figure to the front.

        Args:
            - fig (plotly.graph_objects.Figure): graph object figure

        """

        label_data = []
        plot_data = []
        for data in fig.data:
            if data.mode == 'text':
                label_data.append(data)
            else:
                plot_data.append(data)

        fig.data = tuple(plot_data + label_data)

    @staticmethod
    def create_geometry_figure(ndim: int,
                               geometry: 'Geometry',
                               show_volume_ids: bool = False,
                               show_surface_ids: bool = False,
                               show_line_ids: bool = False,
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
            PlotUtils.__add_2d_lines_to_plot(geometry, show_line_ids, show_point_ids, fig)

            # loop over all surfaces
            for surface in geometry.surfaces.values():
                PlotUtils.__add_2d_surface_to_plot(geometry, surface, show_surface_ids, fig)

            # reorder data such that all text labels are shown
            PlotUtils.__move_labels_to_front(fig)

        elif ndim == 3:

            PlotUtils.__add_3d_lines_to_plot(geometry, show_line_ids, show_point_ids, fig)
            # loop over all volumes
            for volume_data in geometry.volumes.values():

                PlotUtils.__add_3d_volume_to_plot(geometry, volume_data, show_volume_ids, show_surface_ids, fig)

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

            # yapf: disable
            fig.update_layout(xaxis={"title": "x-coordinates [m]",
                                     "range": xlim},
                              yaxis={"title": "y-coordinates [m]",
                                     "range": ylim},
                              showlegend=False,
                              hovermode=False)
            # yapf: enable

        elif ndim == 3:

            # calculate and set z limits
            min_z, max_z = np.min(all_coordinates[:, 2]), np.max(all_coordinates[:, 2])
            dz = max_z - min_z

            zlim = [min_z - buffer * dz, max_z + buffer * dz]

            # yapf: disable
            scene = dict(xaxis={"title": "x-coordinates [m]",
                                "range": xlim},
                         yaxis={"title": "y-coordinates [m]",
                                "range": ylim},
                         zaxis={"title": "z-coordinates [m]",
                                "range": zlim},
                         camera={"up": {"x": 0, "y": 1, "z": 0}})
            # yapf: enable
            fig.update_layout(scene=scene)

        else:
            raise ValueError("Number of dimensions should be 2 or 3")

        # set layout
        fig.update_layout(showlegend=False, hovermode=False)

        return fig
