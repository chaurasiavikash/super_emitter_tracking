# ============================================================================
# FILE: src/utils/atmospheric_utils.py
# ============================================================================
import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, Dict
import logging
from scipy import constants
import warnings

logger = logging.getLogger(__name__)

# Physical constants
AVOGADRO = constants.Avogadro  # mol^-1
R_GAS = constants.R  # J/(mol·K)
GRAVITY = 9.80665  # m/s²
M_AIR = 0.02897  # kg/mol (molar mass of dry air)
M_CH4 = 0.01604  # kg/mol (molar mass of methane)
STANDARD_PRESSURE = 101325.0  # Pa
STANDARD_TEMPERATURE = 273.15  # K

def mixing_ratio_to_concentration(mixing_ratio_ppb: Union[float, np.ndarray],
                                 pressure_pa: Union[float, np.ndarray] = STANDARD_PRESSURE,
                                 temperature_k: Union[float, np.ndarray] = STANDARD_TEMPERATURE) -> Union[float, np.ndarray]:
    """
    Convert methane mixing ratio (ppb) to mass concentration (kg/m³).
    
    Args:
        mixing_ratio_ppb: Methane mixing ratio in parts per billion
        pressure_pa: Atmospheric pressure in Pascal
        temperature_k: Temperature in Kelvin
        
    Returns:
        Mass concentration in kg/m³
    """
    
    # Convert ppb to mole fraction
    mole_fraction = mixing_ratio_ppb * 1e-9
    
    # Calculate air density using ideal gas law
    air_density = (pressure_pa * M_AIR) / (R_GAS * temperature_k)  # kg/m³
    
    # Calculate CH4 mass concentration
    ch4_concentration = mole_fraction * air_density * (M_CH4 / M_AIR)  # kg/m³
    
    return ch4_concentration

def concentration_to_mixing_ratio(concentration_kg_m3: Union[float, np.ndarray],
                                 pressure_pa: Union[float, np.ndarray] = STANDARD_PRESSURE,
                                 temperature_k: Union[float, np.ndarray] = STANDARD_TEMPERATURE) -> Union[float, np.ndarray]:
    """
    Convert methane mass concentration (kg/m³) to mixing ratio (ppb).
    
    Args:
        concentration_kg_m3: Mass concentration in kg/m³
        pressure_pa: Atmospheric pressure in Pascal
        temperature_k: Temperature in Kelvin
        
    Returns:
        Mixing ratio in parts per billion
    """
    
    # Calculate air density
    air_density = (pressure_pa * M_AIR) / (R_GAS * temperature_k)  # kg/m³
    
    # Calculate mole fraction
    mole_fraction = concentration_kg_m3 / air_density * (M_AIR / M_CH4)
    
    # Convert to ppb
    mixing_ratio_ppb = mole_fraction * 1e9
    
    return mixing_ratio_ppb

def column_to_mixing_ratio(column_density_kg_m2: Union[float, np.ndarray],
                          surface_pressure_pa: Union[float, np.ndarray] = STANDARD_PRESSURE) -> Union[float, np.ndarray]:
    """
    Convert column density (kg/m²) to column-averaged mixing ratio (ppb).
    
    Args:
        column_density_kg_m2: Column density in kg/m²
        surface_pressure_pa: Surface pressure in Pascal
        
    Returns:
        Column-averaged mixing ratio in ppb
    """
    
    # Total air column mass (kg/m²)
    air_column_mass = surface_pressure_pa / GRAVITY
    
    # Mass ratio
    mass_ratio = column_density_kg_m2 / air_column_mass
    
    # Convert to mole ratio and then to ppb
    mole_ratio = mass_ratio * (M_AIR / M_CH4)
    mixing_ratio_ppb = mole_ratio * 1e9
    
    return mixing_ratio_ppb

def mixing_ratio_to_column(mixing_ratio_ppb: Union[float, np.ndarray],
                          surface_pressure_pa: Union[float, np.ndarray] = STANDARD_PRESSURE) -> Union[float, np.ndarray]:
    """
    Convert column-averaged mixing ratio (ppb) to column density (kg/m²).
    
    Args:
        mixing_ratio_ppb: Column-averaged mixing ratio in ppb
        surface_pressure_pa: Surface pressure in Pascal
        
    Returns:
        Column density in kg/m²
    """
    
    # Convert ppb to mole fraction
    mole_fraction = mixing_ratio_ppb * 1e-9
    
    # Total air column mass (kg/m²)
    air_column_mass = surface_pressure_pa / GRAVITY
    
    # Calculate CH4 column density
    ch4_column_density = mole_fraction * air_column_mass * (M_CH4 / M_AIR)
    
    return ch4_column_density

def calculate_air_mass_factor(solar_zenith_angle_deg: Union[float, np.ndarray],
                             viewing_zenith_angle_deg: Union[float, np.ndarray] = 0.0) -> Union[float, np.ndarray]:
    """
    Calculate air mass factor for satellite observations.
    
    Args:
        solar_zenith_angle_deg: Solar zenith angle in degrees
        viewing_zenith_angle_deg: Satellite viewing zenith angle in degrees
        
    Returns:
        Air mass factor (dimensionless)
    """
    
    # Convert to radians
    sza_rad = np.radians(solar_zenith_angle_deg)
    vza_rad = np.radians(viewing_zenith_angle_deg)
    
    # Simple geometric air mass factor calculation
    # More sophisticated calculations would include atmospheric profile effects
    amf = 1.0 / np.cos(sza_rad) + 1.0 / np.cos(vza_rad)
    
    # Clip to reasonable values
    amf = np.clip(amf, 1.0, 10.0)
    
    return amf

def estimate_boundary_layer_height(temperature_k: Union[float, np.ndarray],
                                  surface_pressure_pa: Union[float, np.ndarray],
                                  wind_speed_ms: Union[float, np.ndarray],
                                  time_of_day_hours: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Estimate atmospheric boundary layer height using empirical relationships.
    
    Args:
        temperature_k: Surface temperature in Kelvin
        surface_pressure_pa: Surface pressure in Pascal
        wind_speed_ms: Wind speed in m/s
        time_of_day_hours: Time of day in hours (0-24)
        
    Returns:
        Estimated boundary layer height in meters
    """
    
    # Base height from temperature (warmer = higher BL)
    base_height = 200.0 + (temperature_k - 273.15) * 20.0  # meters
    
    # Wind speed effect (more mixing = higher BL)
    wind_factor = 1.0 + 0.1 * wind_speed_ms
    
    # Diurnal cycle (higher during day)
    diurnal_factor = 1.0 + 0.5 * np.sin(np.pi * (time_of_day_hours - 6) / 12)
    diurnal_factor = np.maximum(diurnal_factor, 0.5)  # Minimum factor
    
    # Pressure effect (lower pressure = higher altitude = higher BL)
    pressure_factor = STANDARD_PRESSURE / surface_pressure_pa
    
    # Combined estimate
    bl_height = base_height * wind_factor * diurnal_factor * pressure_factor
    
    # Clip to reasonable range
    bl_height = np.clip(bl_height, 100.0, 3000.0)
    
    return bl_height

def calculate_wind_correction_factor(wind_speed_ms: Union[float, np.ndarray],
                                   wind_direction_deg: Union[float, np.ndarray],
                                   satellite_overpass_direction_deg: Union[float, np.ndarray] = 0.0) -> Union[float, np.ndarray]:
    """
    Calculate wind correction factor for emission estimates.
    
    Args:
        wind_speed_ms: Wind speed in m/s
        wind_direction_deg: Wind direction in degrees (meteorological convention)
        satellite_overpass_direction_deg: Satellite overpass direction in degrees
        
    Returns:
        Wind correction factor (dimensionless)
    """
    
    # Convert wind direction to radians
    wind_dir_rad = np.radians(wind_direction_deg)
    overpass_dir_rad = np.radians(satellite_overpass_direction_deg)
    
    # Calculate relative angle
    relative_angle = wind_dir_rad - overpass_dir_rad
    
    # Correction factor based on wind speed and direction relative to satellite track
    # Stronger winds and perpendicular directions lead to better dispersion
    speed_factor = np.minimum(wind_speed_ms / 5.0, 2.0)  # Normalize to typical wind speed
    direction_factor = np.abs(np.sin(relative_angle))  # Maximum for perpendicular wind
    
    correction_factor = 1.0 + 0.5 * speed_factor * direction_factor
    
    return correction_factor

def estimate_emission_rate_simple(enhancement_ppb: Union[float, np.ndarray],
                                 area_km2: Union[float, np.ndarray],
                                 wind_speed_ms: Union[float, np.ndarray] = 5.0,
                                 boundary_layer_height_m: Union[float, np.ndarray] = 1000.0) -> Union[float, np.ndarray]:
    """
    Simple emission rate estimation using box model approach.
    
    This is a simplified approach for demonstration. Real emission quantification
    requires sophisticated atmospheric transport modeling.
    
    Args:
        enhancement_ppb: Methane enhancement in ppb
        area_km2: Source area in km²
        wind_speed_ms: Wind speed in m/s
        boundary_layer_height_m: Boundary layer height in m
        
    Returns:
        Estimated emission rate in kg/hr
    """
    
    logger.warning("Using simplified emission estimation. For accurate quantification, use atmospheric transport models.")
    
    # Convert enhancement to mass concentration
    mass_conc = mixing_ratio_to_concentration(enhancement_ppb)  # kg/m³
    
    # Convert area to m²
    area_m2 = area_km2 * 1e6
    
    # Simple box model: emission = concentration × volume flow rate
    # Volume flow rate = area × wind speed × mixing height
    volume_flow_rate = area_m2 * wind_speed_ms * boundary_layer_height_m  # m³/s