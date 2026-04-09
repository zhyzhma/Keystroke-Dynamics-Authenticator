from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_env_file = "../.env"

class Settings(BaseSettings):
    ALGORITHM: str = Field(max_length=30)

    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(ge=1, le=60 * 24 * 30)


    SECRET_KEY: str = Field(max_length=100)

    #DATABASE_URL: str = Field()

    SERVER_PORT: int = Field(ge=0, le=9999)
    SERVER_IP: str = Field() # ?

    DATABASE_PREFIX: str = Field()
    DATABASE_USER: str = Field() 
    POSTGRES_PASSWORD: str = Field()
    DATABASE_HOST: str = Field()
    DATABASE_PORT: int = Field(le=9999)
    DATABASE_NAME: str = Field()
    EXTERNAL_DATABASE_PORT: int = Field(le=9999)

    model_config = SettingsConfigDict(env_file=_env_file, env_file_encoding="utf-8")

env_settings: Settings = Settings()
