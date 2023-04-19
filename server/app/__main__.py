import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.core.config.config import MarlConfig

with open("MarlConfig.json", "w") as f:
    f.write(MarlConfig().json(indent=4))
