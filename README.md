# Multi-Sensor Data Pipeline for Robotics Experiments

A web-based application for recording, cleaning, and synchronizing data from multiple sensors used in robotics experiments. Built to reduce alignment errors and simplify analysis for later experiments.

## Features

- **Data Input**: Generate sample sensor data or upload custom CSV files
- **Data Cleaning**: Remove noise, handle missing values, fix outliers
- **Synchronization**: Align data from multiple sensors to a common timeline
- **Visualization**: Interactive charts showing sensor data and correlations
- **Export**: Download synchronized dataset as CSV

## Supported Sensors

| Sensor Type | Data Captured |
|-------------|---------------|
| Camera | Object tracking (X, Y positions, size, confidence) |
| Motion/IMU | Accelerometer (X, Y, Z), Gyroscope (X, Y, Z) |
| External Logs | Gripper events, arm movements, calibration events |

## Installation

### Prerequisites
- Python 3.11+
- pip or uv package manager

### Setup

1. Clone the repository:
bash
git clone https://github.com/yourusername/multi-sensor-data-pipeline.git
cd multi-sensor-data-pipeline


2. Install dependencies:
pip install -r requirements.txt

3. Run the application:
streamlit run app.py --server.port 5000


4. Open your browser to `http://localhost:5000`

## Usage

1. Load Data: Click "Generate Sample Dataset" to create test data, or upload your own CSV files
2. Clean Data: Use the Data Cleaning tab to remove noise and fix errors
3. Synchronize: Align all sensors to a common timeline
4. Visualize: Explore the synchronized data with interactive charts
5. Export: Download the processed data for further analysis

## Project Structure

```
multi-sensor-data-pipeline/
 app.py                  Main Streamlit application
 requirements.txt        Python dependencies
 README.md              
 streamlit/
       config.toml       Streamlit configuration

## Technologies Used

- **Streamlit** - Web application framework
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Plotly** - Interactive visualizations

## Author

**Sriram M**
- Email: sriram162003@icloud.com
- Education: B.Tech in Computer Science Engineering, SRM University, Trichy (2023-2027)

## License

This project is open source and available under the MIT License.
