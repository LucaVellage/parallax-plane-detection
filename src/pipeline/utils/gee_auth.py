"""
Script for initial authentication and initialisation of Google Earth Engine
"""

import ee

from pipeline.settings import get_settings

def gee_auth_init():
    settings = get_settings()
    ee.Authenticate()
    ee.Initialize(project=settings.gee_project)
    print(f'GEE initialised successfully')
