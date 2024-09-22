import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, BaseModel

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir))


class TinyImageNet(BaseModel):
    TRAIN_FILE: str
    VALID_FILE: str  # Validation File


class StanfordCars(BaseModel):
    TRAIN_FILE: str
    TEST_FILE: str  # Validation File

    WIDTH: int = 360
    HEIGHT: int = 240


class Settings(BaseSettings):

    APPLICATION_NAME: str = "Computer Vision Fun"

    DATA_FOLDER: str
    OUT_FOLDER: str

    TINY_IMAGENET: TinyImageNet
    STANFORD_CARS: StanfordCars

    class Config:
        case_sensitive = True
        env_file_encoding = "utf-8"
        env_file = os.path.join(ENV_FILE_DIR, '.env')
        env_nested_delimiter = '__'
        extra='allow'


class Constants(BaseSettings):
    DATE_TIME_UTC: str = 'datetime_utc'
    ID: str = 'id'


class ModelSettings(BaseSettings):
    device: str = "cuda"


settings = Settings()
model_settings = ModelSettings()
