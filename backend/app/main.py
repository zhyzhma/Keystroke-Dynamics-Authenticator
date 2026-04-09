from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import dbase

from app.presentation import security

from app.settings import env_settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.sessions = {}
    
    # Migration if needed
    await dbase.migrate_models()

    try:
        yield
    finally:
        await dbase.async_engine.dispose()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(security.security_route)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=env_settings.SERVER_IP, port=env_settings.SERVER_PORT)