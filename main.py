#!/usr/bin/env python3
"""Entry point: python main.py [--help]"""

import sys

from gesture_controller.app import main

if __name__ == "__main__":
    sys.exit(main())
