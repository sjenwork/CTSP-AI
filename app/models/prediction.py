from typing import List, Optional

from pydantic import BaseModel


class TimeSeriesData(BaseModel):
    TimePoint: str
    PM25: Optional[float]
    PM10: Optional[float]
    P: Optional[float]
    WS: Optional[float]
    WD_sin: Optional[float]
    WD_cos: Optional[float]
    CH4: Optional[float]
    CO: Optional[float]
    NMHC: Optional[float]
    NO: Optional[float]
    NO2: Optional[float]
    NOx: Optional[float]
    O3: Optional[float]
    SO2: Optional[float]
    TimeIndex: int


class MeanData(BaseModel):
    PM25: Optional[float]
    PM10: Optional[float]
    P: Optional[float]
    WS: Optional[float]
    WD_sin: Optional[float]
    WD_cos: Optional[float]
    CH4: Optional[float]
    CO: Optional[float]
    NMHC: Optional[float]
    NO: Optional[float]
    NO2: Optional[float]
    NOx: Optional[float]
    O3: Optional[float]
    SO2: Optional[float]


class StdData(BaseModel):
    PM25: Optional[float]
    PM10: Optional[float]
    P: Optional[float]
    WS: Optional[float]
    WD_sin: Optional[float]
    WD_cos: Optional[float]
    CH4: Optional[float]
    CO: Optional[float]
    NMHC: Optional[float]
    NO: Optional[float]
    NO2: Optional[float]
    NOx: Optional[float]
    O3: Optional[float]
    SO2: Optional[float]


class PredictorData(BaseModel):
    TimeSeries: List[TimeSeriesData]
    Mean: MeanData
    Std: StdData
