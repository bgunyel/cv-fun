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
    NUM_CLASSES: int = 196


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


class InceptionV3(BaseModel):
    MODEL_NAME: str = 'timm/inception_v3.tf_adv_in1k'
    INPUT_DIMS: tuple = (3, 299, 299)
    FEATURE_MAP_DIMS: tuple = (2048, 8, 8)
    HIDDEN_LAYER_SIZE: int = 256
    DROP_OUT_PROB: float = 0.5


class ModelSettings(BaseSettings):
    DEVICE: str = "cuda"
    INCEPTION_V3: InceptionV3 = InceptionV3()


class Constants(BaseSettings):
    DATE_TIME_UTC: str = 'datetime_utc'
    ID: str = 'id'


settings = Settings()
model_settings = ModelSettings()
