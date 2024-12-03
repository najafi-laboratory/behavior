# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 21:39:16 2024

@author: gtg424h
"""

import pandas as pd

xml_data = '.\\FN14_20240611_seq1420_t-002.xml'


# Load the XML into a pandas DataFrame
df = pd.read_xml(xml_data)

# Show the resulting DataFrame
# print(df)

print(df.tail())