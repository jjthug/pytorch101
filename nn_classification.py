import sklearn

from sklearn.datasets import make_circles

n_sample = 1000

x,y = make_circles(n_sample,
                   noise = 0.03,
                   random_state=69)

print(f"length of x => {len(x)}, length of y = {len(y)}")

print(f"first 5 samples of x => {x[:5]}, first 5 samples of y => {y[:10]}")

import pandas as pd
circles = pd.DataFrame({"X1": x[:, 0],
                        "X2": x[:, 1],
                        "label": y})

circles.head(10)