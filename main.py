import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "gin rummy"))
sys.path.append(os.path.join(os.path.dirname(__file__), "model"))

from gui import GinRummyGUI

if __name__ == "__main__":
    app = GinRummyGUI()
    app.run()
