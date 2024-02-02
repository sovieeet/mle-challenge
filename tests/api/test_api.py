import unittest
from unittest.mock import patch, ANY
from fastapi.testclient import TestClient
from challenge.api import PredictionInput
from challenge import app  # Importa la aplicación FastAPI desde tu módulo principal

class TestBatchPipeline(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)  # Utiliza la aplicación FastAPI como argumento

    @patch('challenge.model.DelayModel.predict')
    def test_should_get_predict(self, mock_predict):
        mock_predict.return_value = [0]
        data = {
            "OPERA": "Aerolineas Argentinas",
            "TIPOVUELO": "N",
            "MES": "3",  # Asegúrate de que MES sea una cadena (string)
            "Fecha_I": "2024-01-01 12:00:00",  # Ajusta la fecha y hora según tus necesidades
            "Fecha_O": "2024-01-01 14:00:00"
        }
        input_data = PredictionInput(**data)
        response = self.client.post("/predict", json=input_data.dict())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"prediction": None})

    def test_should_failed_unkown_column_1(self):
        data = {       
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas", 
                    "TIPOVUELO": "N",
                    "MES": 13
                }
            ]
        }
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 422)

    def test_should_failed_unkown_column_2(self):
        data = {        
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas", 
                    "TIPOVUELO": "O", 
                    "MES": 13
                }
            ]
        }
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 422)
    
    def test_should_failed_unkown_column_3(self):
        data = {        
            "flights": [
                {
                    "OPERA": "Argentinas", 
                    "TIPOVUELO": "O", 
                    "MES": 13
                }
            ]
        }
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 422)