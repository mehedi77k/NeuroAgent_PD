from pydantic import BaseSettings


class Settings(BaseSettings):
    app_name: str = "NeuroAgent PD"
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000


settings = Settings()
