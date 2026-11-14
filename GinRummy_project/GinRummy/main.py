import os
import sys

# Ensure local submodules are cleanly discoverable
sys.path.append(os.path.join(os.path.dirname(__file__), "gin rummy"))
sys.path.append(os.path.join(os.path.dirname(__file__), "model"))

from gui import GinRummyGUI

if __name__ == "__main__":
    app = GinRummyGUI()
    app.run()