"""
Tests for the iwp_tracking.collocations module.
"""
from iwp_tracking.collocations import get_equal_area_grid


def test_get_equal_area_grid():
    """
    Create an equal area grid centered on lon = 0.0, lat = 0.0 and ensure that
    those coordinates are mapped to the center of the grid.
    """
    lon = 0.0
    lat = 0.0
    area = get_equal_area_grid(lon, lat, resolution=5e3, extent=500e3)
    x, y = area.get_array_coordinates_from_lonlat(lon, lat)
    assert x == 49.5
    assert y == 49.5
