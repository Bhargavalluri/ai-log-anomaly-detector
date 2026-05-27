from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PORT: int = 8000
    APP_NAME: str = "AI Log Anomaly Detector"

    class Config:
        env_file = ".env"

settings = Settings()
