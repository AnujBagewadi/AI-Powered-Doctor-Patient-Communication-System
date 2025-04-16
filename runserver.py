"""
This script runs the Doctor_Patient_communication_system application using a development server.
"""

from os import environ
from Doctor_Patient_communication_system import app

if __name__ == '__main__':
    HOST = environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.run(HOST, PORT)
