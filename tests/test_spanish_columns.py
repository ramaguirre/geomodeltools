"""Tables with Spanish column names (Este/Norte/Cota, Sondaje, Desde/Hasta...) are found as the English ones are."""
import numpy as np
import pandas as pd

from geomodeltools.columns import XYZ_CANDIDATES, find_columns
from geomodeltools.desurvey import desurvey_intervals


def test_este_norte_cota_are_coordinates_in_any_case():
    for cols in (["Este", "Norte", "Cota"], ["ESTE", "NORTE", "ELEVACION"], ["este", "norte", "elevación"], ["ESTE", "NORTE", "Z"]):
        df = pd.DataFrame(np.zeros((2, 3)), columns=cols)
        assert find_columns(df, XYZ_CANDIDATES) == tuple(cols)


def test_a_drillhole_set_in_spanish_desurveys_as_the_english_one():
    en_c = pd.DataFrame({"HOLEID": ["A", "B"], "X": [100.0, 200.0], "Y": [50.0, 60.0], "Z": [400.0, 410.0], "LENGTH": [80.0, 60.0]})
    en_s = pd.DataFrame({"HOLEID": ["A", "A", "B"], "DEPTH": [0.0, 80.0, 0.0], "AZIMUTH": [90.0, 95.0, 180.0], "DIP": [-60.0, -62.0, -90.0]})
    en_i = pd.DataFrame({"HOLEID": ["A", "A", "B"], "FROM": [0.0, 30.0, 0.0], "TO": [30.0, 80.0, 60.0], "CU": [0.5, 0.8, 0.3]})
    es_c = en_c.rename(columns={"HOLEID": "Sondaje", "X": "Este", "Y": "Norte", "Z": "Cota", "LENGTH": "Longitud"})
    es_s = en_s.rename(columns={"HOLEID": "SONDAJE", "DEPTH": "PROFUNDIDAD", "AZIMUTH": "AZIMUT", "DIP": "INCLINACION"})
    es_i = en_i.rename(columns={"HOLEID": "SONDAJE", "FROM": "DESDE", "TO": "HASTA"})
    en = desurvey_intervals(en_c, en_s, en_i, dip_positive_down=False).intervals
    es = desurvey_intervals(es_c, es_s, es_i, dip_positive_down=False).intervals
    for c in ("start_x", "start_y", "start_z", "mid_x", "mid_y", "mid_z", "end_x", "end_y", "end_z"):
        np.testing.assert_allclose(es[c], en[c])
