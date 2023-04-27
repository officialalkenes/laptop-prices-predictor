import pickle

import numpy as np
import streamlit as st


# Load Processed model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))
