from app.db.engine import get_engine, Base
from app.db.models.models import Model
from app.db.models.providers import Provider
from app.db.models.video_tasks import VideoTask

def init_db():
    # Import model classes before create_all so their tables are registered.
    _ = (Model, Provider, VideoTask)
    engine = get_engine()

    Base.metadata.create_all(bind=engine)
