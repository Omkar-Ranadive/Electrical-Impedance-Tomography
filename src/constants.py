"""
This file will contain constants used throughout the project
"""

from pathlib import Path


# Directory paths
PARENT_PATH = Path(__file__).parent
DATA_PATH = PARENT_PATH / '../data'
EXP_PATH = PARENT_PATH / '../exp'


# Matrix dimensions of the form (C: Contacts, M: Polys, D: Measurements)
data_list = [
(4, 6, 2),
(5, 5, 5),
(6, 14, 9),
(7, 14, 14),
(8, 28, 20),
(9, 27, 27),
(11, 44, 44),
(13, 65, 65),
(16, 120, 104),
(6, 6, 2),
(8, 5, 5),
(9, 15, 9),
(11, 14, 14),
(13, 28, 20),
(15, 27, 27),
(18, 44, 44),
(22, 65, 65),
(27, 120, 104),
(7, 6, 2),
(9, 15, 5),
(12, 14, 9),
(14, 14, 14),
(16, 28, 20),
(18, 27, 27),
(23, 44, 44),
(28, 65, 65),
(34, 120, 104),
(8, 6, 2),
(12, 5, 5),
(15, 15, 9),
(19, 14, 14),
(22, 28, 20),
(25, 27, 27),
(32, 44, 44),
(39, 65, 65),
(48, 120, 104)
]