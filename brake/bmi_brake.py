# -*- coding: utf-8 -*-
"""
Created on Fri May 26 09:19:55 2017

@author: Charlie

Basic Model Interface implementation for BRaKE
26 May 2017-- CSDMS BMI hackathon
"""

import types
import numpy as np
import yaml
from basic_modeling_interface import Bmi

from .core import Brake


class BmiBrake(Bmi):
    """
    Calculate river incision and block dynamics on channel long profile
    """
    
    _name = 'The Blocky River and Knickpoint Evolution model'
    _input_var_names = ('topographic__elevation',)
    _output_var_names = ('topographic__elevation',
                        'topographic__steepest_slope',
                        'surface_water__discharge',
                        'channel_water__mean_depth',
                        'channel_bottom_water_flow__speed',
                        'channel_bottom_water_flow__dimensionless_drag_stress',
                        'channel_bottom_water_flow__magnitude_of_shear_stress',
                        'channel_bottom_water_flow__magnitude_of_available_shear_stress',
                        'bedrock__time_averaged_incision_rate',
                        'channel__total_blocks',
                        'blocks__side_length_0_to_1_meters',
                        'blocks__side_length_1_to_2_meters',
                        'blocks__side_length_2_to_3_meters',
                        'blocks__side_length_3_to_4_meters',
                        'blocks__side_length_4_to_5_meters',
                        'blocks__side_length_5_to_6_meters',
                        'blocks__side_length_6_to_7_meters',
                        'blocks__side_length_7_to_8_meters',
                        'blocks__side_length_8_to_9_meters',
                        'blocks__side_length_9_to_10_meters',)
    
    def __init__(self):
        """Create a BmiBrake model that is ready for initialization."""
        self._brake_eroder = None
        self._values = {}
        self._var_units = {}
        self._grids = {}
        self._grid_type = {}

    def initialize(self, filename=None):
        """Initialize the BRaKE model.
        Parameters
        ----------
        filename : str, optional
            Path to name of input file.
        Examples
        --------
        >>> from brake.bmi_brake import BmiBrake
        >>> eroder = BmiBrake()
        >>> print eroder.get_component_name()
        The Blocky River and Knickpoint Evolution model
        >>> print eroder.get_input_var_names()[0]
        topographic__elevation
        >>> print eroder.get_output_var_names()[0]
        topographic__elevation
        >>> eroder.initialize('brake.yaml')
        >>> print eroder.get_start_time()
        0.0
        >>> print eroder.get_current_time()
        0.0
        >>> print eroder.get_time_step()
        10.0
        >>> print eroder.get_time_units()
        year
        >>> topo =  eroder.get_value('topographic__elevation')
        >>> print topo #doctest: +NORMALIZE_WHITESPACE
        [ 10.    9.9   9.8   9.7   9.6   9.5   9.4   9.3   9.2   9.1   9.    8.9
        8.8   8.7   8.6   8.5   8.4   8.3   8.2   8.1   8.    7.9   7.8   7.7
        7.6   7.5   7.4   7.3   7.2   7.1   7.    6.9   6.8   6.7   6.6   6.5
        6.4   6.3   6.2   6.1   6.    5.9   5.8   5.7   5.6   5.5   5.4   5.3
        5.2   5.1   5.    4.9   4.8   4.7   4.6   4.5   4.4   4.3   4.2   4.1
        4.    3.9   3.8   3.7   3.6   3.5   3.4   3.3   3.2   3.1   3.    2.9
        2.8   2.7   2.6   2.5   2.4   2.3   2.2   2.1   2.    1.9   1.8   1.7
        1.6   1.5   1.4   1.3   1.2   1.1   1.    0.9   0.8   0.7   0.6   0.5
        0.4   0.3   0.2   0.1]
        >>> topo[0] = 11.11
        >>> eroder.set_value('topographic__elevation', topo)
        >>> topo = eroder.get_value('topographic__elevation')
        >>> print topo[0]
        11.11
        >>> for dt in range(100):
        ...     eroder.update()
        >>> topo = eroder.get_value('topographic__elevation')
        >>> print topo #doctest: +NORMALIZE_WHITESPACE
        [ 11.05984798   9.89175222   9.79175222   9.69175222   9.59175222
        9.49175222   9.39175222   9.29175222   9.19175222   9.09175222
        8.99175222   8.89175222   8.79175222   8.69175222   8.59175222
        8.49175222   8.39175222   8.29175222   8.19175222   8.09175222
        7.99175222   7.89175222   7.79175222   7.69175222   7.59175222
        7.49175222   7.39175222   7.29175222   7.19175222   7.09175222
        6.99175222   6.89175222   6.79175222   6.69175222   6.59175222
        6.49175222   6.39175222   6.29175222   6.19175222   6.09175222
        5.99175222   5.89175222   5.79175222   5.69175222   5.59175222
        5.49175222   5.39175222   5.29175222   5.19175222   5.09175222
        4.99175222   4.89175222   4.79175222   4.69175222   4.59175222
        4.49175222   4.39175222   4.29175222   4.19175222   4.09175222
        3.99175222   3.89175222   3.79175222   3.69175222   3.59175222
        3.49175222   3.39175222   3.29175222   3.19175222   3.09175222
        2.99175222   2.89175222   2.79175222   2.69175222   2.59175222
        2.49175222   2.39175222   2.29175222   2.19175222   2.09175222
        1.99175222   1.89175222   1.79175222   1.69175222   1.59175222
        1.49175222   1.39175222   1.29175222   1.19175222   1.09175222
        0.99175222   0.89175222   0.79175222   0.69175222   0.59175222
        0.49175221   0.39175185   0.29172919   0.19061605   0.05      ]
        >>> eroder.finalize()
        """
        self._brake_eroder = Brake(config_file=filename)

        self._values = {
            'topographic__elevation': self._brake_eroder.surface_elev_array,
            'topographic__steepest_slope': self._brake_eroder.slope,
            'surface_water__discharge': self._brake_eroder.q,
            'channel_water__mean_depth': self._brake_eroder.flow_depth,
            'channel_bottom_water_flow__speed': self._brake_eroder.flow_velocity,
            'channel_bottom_water_flow__dimensionless_drag_stress': self._brake_eroder.sigma_d_array,
            'channel_bottom_water_flow__magnitude_of_shear_stress': self._brake_eroder.uncorrected_tau_array,
            'channel_bottom_water_flow__magnitude_of_available_shear_stress': self._brake_eroder.corrected_tau_array,
            'bedrock__time_averaged_incision_rate': self._brake_eroder.time_avg_inc_rate_array,
            'channel__total_blocks': self._brake_eroder.blocks_in_cells,
            'blocks__side_length_0_to_1_meters': self._brake_eroder.zero_to_one_temp,
            'blocks__side_length_1_to_2_meters': self._brake_eroder.one_to_two_temp,
            'blocks__side_length_2_to_3_meters': self._brake_eroder.two_to_three_temp,
            'blocks__side_length_3_to_4_meters': self._brake_eroder.three_to_four_temp,
            'blocks__side_length_4_to_5_meters': self._brake_eroder.four_to_five_temp,
            'blocks__side_length_5_to_6_meters': self._brake_eroder.five_to_six_temp,
            'blocks__side_length_6_to_7_meters': self._brake_eroder.six_to_seven_temp,
            'blocks__side_length_7_to_8_meters': self._brake_eroder.seven_to_eight_temp,
            'blocks__side_length_8_to_9_meters': self._brake_eroder.eight_to_nine_temp,
            'blocks__side_length_9_to_10_meters': self._brake_eroder.nine_to_ten_temp,
        }
        self._var_units = {
            'topographic__elevation': 'm',
            'topographic__steepest_slope': '-',
            'surface_water__discharge': 'm2/s',
            'channel_water__mean_depth': 'm',
            'channel_bottom_water_flow__speed': 'm/s',
            'channel_bottom_water_flow__dimensionless_drag_stress': '-',
            'channel_bottom_water_flow__magnitude_of_shear_stress': 'Pa',
            'channel_bottom_water_flow__magnitude_of_available_shear_stress': 'Pa',
            'bedrock__time_averaged_incision_rate': 'm/yr',
            'channel__total_blocks': '#',
            'blocks__side_length_0_to_1_meters': '#',
            'blocks__side_length_1_to_2_meters': '#',
            'blocks__side_length_2_to_3_meters': '#',
            'blocks__side_length_3_to_4_meters': '#',
            'blocks__side_length_4_to_5_meters': '#',
            'blocks__side_length_5_to_6_meters': '#',
            'blocks__side_length_6_to_7_meters': '#',
            'blocks__side_length_7_to_8_meters': '#',
            'blocks__side_length_8_to_9_meters': '#',
            'blocks__side_length_9_to_10_meters': '#',
        }
        self._grids = {
            0: ['topographic__elevation', 'topographic__steepest_slope',
                'surface_water__discharge', 'channel_water__mean_depth',
                'channel_bottom_water_flow__speed', 
                'channel_bottom_water_flow__dimensionless_drag_stress',
                'channel_bottom_water_flow__magnitude_of_shear_stress',
                'channel_bottom_water_flow__magnitude_of_available_shear_stress',
                'bedrock__time_averaged_incision_rate',
                'channel__total_blocks',
                'blocks__side_length_0_to_1_meters',
                'blocks__side_length_1_to_2_meters',
                'blocks__side_length_2_to_3_meters',
                'blocks__side_length_3_to_4_meters',
                'blocks__side_length_4_to_5_meters',
                'blocks__side_length_5_to_6_meters',
                'blocks__side_length_6_to_7_meters',
                'blocks__side_length_7_to_8_meters',
                'blocks__side_length_8_to_9_meters',
                'blocks__side_length_9_to_10_meters',
                ]
        }
        self._grid_type = {
            0: 'uniform_profile'
        }

    def update(self):
        """Advance model by one time step."""
        self._brake_eroder.update()

    def update_frac(self, time_frac):
        """Update model by a fraction of a time step.
        Parameters
        ----------
        time_frac : float
            Fraction of a time step.
        """
        time_step = self.get_time_step()
        self._brake_eroder.time_step = time_frac * time_step
        self.update()
        self._brake_eroder.time_step = time_step

    def update_until(self, then):
        """Update model until a particular time.
        Parameters
        ----------
        then : float
            Time to run model until.
        """
        n_steps = (then - self.get_current_time()) / self.get_time_step()

        for _ in range(int(n_steps)):
            self.update()
        self.update_frac(n_steps - int(n_steps))

    def finalize(self):
        """Finalize model."""
        self._brake_eroder.finalize()

    def get_var_type(self, var_name):
        """Data type of variable.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        Returns
        -------
        str
            Data type.
        """
        return str(self.get_value_ref(var_name).dtype)

    def get_var_units(self, var_name):
        """Get units of variable.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        Returns
        -------
        str
            Variable units.
        """
        return self._var_units[var_name]

    def get_var_nbytes(self, var_name):
        """Get units of variable.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        Returns
        -------
        int
            Size of data array in bytes.
        """
        return self.get_value_ref(var_name).nbytes

    def get_var_grid(self, var_name):
        """Grid id for a variable.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        Returns
        -------
        int
            Grid id.
        """
        for grid_id, var_name_list in self._grids.items():
            if var_name in var_name_list:
                return grid_id

    def get_grid_rank(self, grid_id):
        """Rank of grid.
        Parameters
        ----------
        grid_id : int
            Identifier of a grid.
        Returns
        -------
        int
            Rank of grid.
        """
        return len(self.get_grid_shape(grid_id))

    def get_grid_size(self, grid_id):
        """Size of grid.
        Parameters
        ----------
        grid_id : int
            Identifier of a grid.
        Returns
        -------
        int
            Size of grid.
        """
        return np.prod(self.get_grid_shape(grid_id))

    def get_value_ref(self, var_name):
        """Reference to values.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        Returns
        -------
        array_like
            Value array.
        """
        return self._values[var_name]

    def get_value(self, var_name):
        """Copy of values.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        Returns
        -------
        array_like
            Copy of values.
        """
        return self.get_value_ref(var_name).copy()

    def get_value_at_indices(self, var_name, indices):
        """Get values at particular indices.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        indices : array_like
            Array of indices.
        Returns
        -------
        array_like
            Values at indices.
        """
        return self.get_value_ref(var_name).take(indices)

    def set_value(self, var_name, src):
        """Set model values.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        src : array_like
            Array of new values.
        """
        val = self.get_value_ref(var_name)
        val[:] = src

    def set_value_at_indices(self, var_name, src, indices):
        """Set model values at particular indices.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        src : array_like
            Array of new values.
        indices : array_like
            Array of indices.
        """
        val = self.get_value_ref(var_name)
        val.flat[indices] = src

    def get_component_name(self):
        """Name of the component."""
        return self._name

    def get_input_var_names(self):
        """Get names of input variables."""
        return self._input_var_names

    def get_output_var_names(self):
        """Get names of output variables."""
        return self._output_var_names

    def get_grid_shape(self, grid_id):
        """Number of rows and columns of uniform rectilinear grid."""
        var_name = self._grids[grid_id][0]
        return self.get_value_ref(var_name).shape

    def get_grid_spacing(self, grid_id):
        """Spacing of rows and columns of uniform rectilinear grid."""
        return self._brake_eroder.spacing

    def get_grid_origin(self, grid_id):
        """Origin of uniform rectilinear grid."""
        return (0., 0.)

    def get_grid_type(self, grid_id):
        """Type of grid."""
        return self._grid_type[grid_id]

    def get_start_time(self):
        """Start time of model."""
        return 0.

    def get_end_time(self):
        """End time of model."""
        return self._brake_eroder.time_to_run

    def get_current_time(self):
        """Current time of model."""
        return self._brake_eroder.time

    def get_time_step(self):
        """Time step of model."""
        return self._brake_eroder.time_step
        
    def get_time_units(self):
        """Time units of model."""
        return 'year'
