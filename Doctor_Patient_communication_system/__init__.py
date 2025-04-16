"""
The flask application package.
"""

from flask import Flask
app = Flask(__name__)

import Doctor_Patient_communication_system.views
