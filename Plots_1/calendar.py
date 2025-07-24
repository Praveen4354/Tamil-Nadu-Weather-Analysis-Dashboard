from plotly_calplot import calplot
import pandas as pa
import numpy as np

df = pa.DataFrame({
  "date": pa.date_range(start="2023-01-01", end="2023-12-31"),
  "value": np.random.randint(1, 10, size=365)
})

fig = calplot(df, x="date", y="value")

fig.show()
