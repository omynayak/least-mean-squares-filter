# Least Mean Squares filter
A simple, *learning based* implementation of the Least Mean Squares filter to remove Motion Artefacts from PPG signal. 

## Features
1. **Real time Adaptive Noise Cancellation**<br>
   Achieved by adapting the weight vector as newer inputs come in:
   ```
   w[n+1] = w[n] + 2μ e[n] x[n]
   ```
   Allowing for automatic adjustment to varying motion conditions.

2. **Complete End-to-End PPG Cleaning Pipeline**<br>
   Goes from loading and reading raw CSV files to computing accelerometer magnitude to adaptive filtering to finally
   filtering and visualisation.

3. **Matplotlib Visualisation**<br>
    Automatically generates mutliple diagnostic plots:
      - Raw PPG
      - Cleaned vs noisy PPG comparison
      - Motion approximation tracking over time
      - 3-axis accelerometer signals

## Project Structure
  ```
.
├── main.py
├── ACC.csv
├── PPG.csv
└── README.md
```

## Build Instructions:
```bash
python3 main.py
```

## How it works
1. **Generate Filter Output** <br>
   The output of the filter at a certain point, denoted y[n] is generated as:<br>
   $y(n) =  w^{T}(n) x(n)$<br>
   Where w(n) is the filter coefficient in the weight vector and x(n) is the motion artefact from the `ACC.csv` file.

2. **Generate Error Output**<br>
   The net error of the system is given by the formula:<br>
   $e(n) = d(n) - y(n)$<br>
   This signal is used as the output (cleaned signal) in this case.

3. **Updating Weight Vector**<br>
   The weight vector is updated as:<br>
   $w(n+1) = w(n) + 2μ e(n) x(n)$<br>
   Where μ is the step size.

## Contributing
Contributions, feedback, and suggestions are welcome! <br>
If you’d like to improve performance, precision, or add benchmarks, feel free to open an issue or PR.
