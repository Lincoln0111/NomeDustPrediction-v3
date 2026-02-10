import sys
import os
from pathlib import Path

project_home = Path(__file__).parent
if str(project_home) not in sys.path:
    sys.path.insert(0, str(project_home))

from app import app as fastapi_app
from a2wsgi import ASGIMiddleware

application = ASGIMiddleware(fastapi_app)

if __name__ == "__main__":
    from wsgiref.simple_server import make_server
    
    print("Starting WSGI server on http://localhost:8000")
    print("Press Ctrl+C to stop")
    
    with make_server('', 8000, application) as httpd:
        httpd.serve_forever()