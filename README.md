# Fire-Weather-Index

/**
* Calculates the Fire-Weather Index (FWI) based on the following parameters: 
* - load dataset
* - Feature_Selection
* - Model_selection
* - Model_evaluation
* - Link_to_flask_file
* - Link_flask_file_to_html_file
*/

```bash
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import requests
import json
import flask

```

