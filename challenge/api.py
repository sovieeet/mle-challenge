import pandas as pd
import numpy as np
from typing import Tuple, Union, List
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import xgboost as xgb
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from typing import List
from typing import Optional

app = FastAPI()

class PredictionInput(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: str
    Fecha_I: str
    Fecha_O: str

class PredictionOutput(BaseModel):
    prediction: Optional[float] = None

class HealthCheckOutput(BaseModel):
    status: str

class DelayModel:

    def __init__(self):
        self._model = xgb.XGBClassifier(random_state=1, learning_rate=0.01)

    def get_period_day(self, date):
        date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
        morning_min = datetime.strptime("05:00", '%H:%M').time()
        morning_max = datetime.strptime("11:59", '%H:%M').time()
        afternoon_min = datetime.strptime("12:00", '%H:%M').time()
        afternoon_max = datetime.strptime("18:59", '%H:%M').time()
        evening_min = datetime.strptime("19:00", '%H:%M').time()
        evening_max = datetime.strptime("23:59", '%H:%M').time()
        night_min = datetime.strptime("00:00", '%H:%M').time()
        night_max = datetime.strptime("4:59", '%H:%M').time()

        if (date_time > morning_min and date_time < morning_max):
            return 'mañana'
        elif (date_time > afternoon_min and date_time < afternoon_max):
            return 'tarde'
        elif (
                (date_time > evening_min and date_time < evening_max) or
                (date_time > night_min and date_time < night_max)
        ):
            return 'noche'

    def is_high_season(self, fecha):
        fecha_año = int(fecha.split('-')[0])
        fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
        range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year=fecha_año)
        range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year=fecha_año)
        range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year=fecha_año)
        range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year=fecha_año)
        range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year=fecha_año)
        range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year=fecha_año)
        range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year=fecha_año)
        range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year=fecha_año)

        if ((fecha >= range1_min and fecha <= range1_max) or
                (fecha >= range2_min and fecha <= range2_max) or
                (fecha >= range3_min and fecha <= range3_max) or
                (fecha >= range4_min and fecha <= range4_max)):
            return 1
        else:
            return 0

    def get_min_diff(self, data):
        fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        min_diff = ((fecha_o - fecha_i).total_seconds()) / 60
        return min_diff

    def preprocess(self, data: pd.DataFrame, target_column: str = None) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        data['period_day'] = data['Fecha-I'].apply(self.get_period_day)
        data['high_season'] = data['Fecha-I'].apply(self.is_high_season)
        data['min_diff'] = data.apply(self.get_min_diff, axis=1)
        
        threshold_in_minutes = 15
        data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)

        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix='OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
            pd.get_dummies(data['MES'], prefix='MES')],
            axis=1
        )

        if target_column:
            return features, data[target_column]
        else:
            return features

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        training_data = shuffle(pd.concat([features, target], axis=1), random_state=111)
        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=42)
        self._model.fit(x_train, y_train)

    def predict(self, features: pd.DataFrame) -> List[int]:
        if self._model is None:
            raise ValueError("Model has not been trained. Please call fit() before predict().")

        xgboost_y_preds = self._model.predict(features)
        return [1 if y_pred > 0.5 else 0 for y_pred in xgboost_y_preds]

delay_model = DelayModel()

@app.get("/health", response_model=HealthCheckOutput, status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", response_model=PredictionOutput, status_code=200)
async def post_predict(input_data: PredictionInput) -> dict:
    try:
        features = delay_model.preprocess(pd.DataFrame([input_data.dict()]))
        prediction = delay_model.predict(features)[0]
        return {"prediction": prediction}
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"prediction": None}
