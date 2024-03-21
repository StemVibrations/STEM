def test_import_kratos():
    """
    Test if KratosMultiphysics and GeoMechanicsApplication can be imported.

    """
    import KratosMultiphysics
    from KratosMultiphysics.GeoMechanicsApplication.geomechanics_analysis import GeoMechanicsAnalysis

    assert KratosMultiphysics is not None
    assert GeoMechanicsAnalysis is not None
