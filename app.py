import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io

st.set_page_config(
    page_title="Multi-Sensor Data Pipeline",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Multi-Sensor Data Pipeline for Robotics Experiments")
st.markdown("**Record, clean, and synchronize data from multiple sensors**")

if 'camera_data' not in st.session_state:
    st.session_state.camera_data = None
if 'motion_data' not in st.session_state:
    st.session_state.motion_data = None
if 'log_data' not in st.session_state:
    st.session_state.log_data = None
if 'synchronized_data' not in st.session_state:
    st.session_state.synchronized_data = None


def generate_sample_camera_data(n_samples=500, start_time=None, freq_hz=30):
    if start_time is None:
        start_time = datetime.now()
    
    time_delta = timedelta(seconds=1/freq_hz)
    timestamps = [start_time + i * time_delta for i in range(n_samples)]
    
    np.random.seed(42)
    noise_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    
    data = {
        'timestamp': timestamps,
        'frame_id': range(n_samples),
        'object_x': np.sin(np.linspace(0, 4*np.pi, n_samples)) * 100 + 200,
        'object_y': np.cos(np.linspace(0, 4*np.pi, n_samples)) * 80 + 150,
        'object_size': np.abs(np.sin(np.linspace(0, 2*np.pi, n_samples))) * 50 + 20,
        'confidence': np.clip(np.random.normal(0.9, 0.1, n_samples), 0, 1)
    }
    
    df = pd.DataFrame(data)
    df.loc[noise_indices, 'object_x'] = np.nan
    df.loc[noise_indices[:len(noise_indices)//2], 'object_y'] = -999
    
    return df


def generate_sample_motion_data(n_samples=600, start_time=None, freq_hz=50):
    if start_time is None:
        start_time = datetime.now() + timedelta(milliseconds=50)
    
    time_delta = timedelta(seconds=1/freq_hz)
    timestamps = [start_time + i * time_delta for i in range(n_samples)]
    
    np.random.seed(43)
    
    data = {
        'timestamp': timestamps,
        'accel_x': np.sin(np.linspace(0, 6*np.pi, n_samples)) + np.random.normal(0, 0.1, n_samples),
        'accel_y': np.cos(np.linspace(0, 6*np.pi, n_samples)) + np.random.normal(0, 0.1, n_samples),
        'accel_z': np.sin(np.linspace(0, 3*np.pi, n_samples)) * 0.5 + 9.8 + np.random.normal(0, 0.05, n_samples),
        'gyro_x': np.sin(np.linspace(0, 4*np.pi, n_samples)) * 2 + np.random.normal(0, 0.2, n_samples),
        'gyro_y': np.cos(np.linspace(0, 4*np.pi, n_samples)) * 2 + np.random.normal(0, 0.2, n_samples),
        'gyro_z': np.sin(np.linspace(0, 2*np.pi, n_samples)) + np.random.normal(0, 0.15, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    spike_indices = np.random.choice(n_samples, size=10, replace=False)
    df.loc[spike_indices, 'accel_x'] = df.loc[spike_indices, 'accel_x'] * 10
    
    return df


def generate_sample_log_data(n_samples=100, start_time=None):
    if start_time is None:
        start_time = datetime.now() - timedelta(milliseconds=100)
    
    np.random.seed(44)
    timestamps = sorted([start_time + timedelta(seconds=np.random.uniform(0, 16)) for _ in range(n_samples)])
    
    event_types = ['GRIPPER_OPEN', 'GRIPPER_CLOSE', 'ARM_MOVE', 'SENSOR_READ', 'CALIBRATION', 'ERROR', 'WARNING']
    events = np.random.choice(event_types, size=n_samples, p=[0.15, 0.15, 0.3, 0.2, 0.1, 0.05, 0.05])
    
    data = {
        'timestamp': timestamps,
        'event_type': events,
        'joint_1': np.random.uniform(-180, 180, n_samples),
        'joint_2': np.random.uniform(-90, 90, n_samples),
        'joint_3': np.random.uniform(-180, 180, n_samples),
        'gripper_force': np.random.uniform(0, 100, n_samples)
    }
    
    return pd.DataFrame(data)


def clean_data(df, sensor_type):
    cleaned = df.copy()
    cleaning_report = []
    
    original_rows = len(cleaned)
    cleaned = cleaned.dropna()
    dropped_na = original_rows - len(cleaned)
    if dropped_na > 0:
        cleaning_report.append(f"Removed {dropped_na} rows with missing values")
    
    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        invalid_mask = (cleaned[col] < -900) | (cleaned[col] > 10000)
        invalid_count = invalid_mask.sum()
        if invalid_count > 0:
            cleaned = cleaned[~invalid_mask]
            cleaning_report.append(f"Removed {invalid_count} outliers from {col}")
    
    if sensor_type == 'motion':
        for col in ['accel_x', 'accel_y', 'gyro_x', 'gyro_y', 'gyro_z']:
            if col in cleaned.columns:
                q1 = cleaned[col].quantile(0.01)
                q99 = cleaned[col].quantile(0.99)
                spike_mask = (cleaned[col] < q1) | (cleaned[col] > q99)
                spike_count = spike_mask.sum()
                if spike_count > 0:
                    cleaned.loc[spike_mask, col] = cleaned[col].median()
                    cleaning_report.append(f"Smoothed {spike_count} spikes in {col}")
    
    if 'timestamp' in cleaned.columns:
        cleaned = cleaned.sort_values('timestamp').reset_index(drop=True)
        cleaning_report.append("Sorted by timestamp")
    
    return cleaned, cleaning_report


def synchronize_sensors(camera_df, motion_df, log_df, method='nearest'):
    if camera_df is None or motion_df is None:
        return None, ["Error: Need at least camera and motion data"]
    
    sync_report = []
    
    camera_df = camera_df.copy()
    motion_df = motion_df.copy()
    
    camera_df['timestamp'] = pd.to_datetime(camera_df['timestamp'])
    motion_df['timestamp'] = pd.to_datetime(motion_df['timestamp'])
    
    camera_df = camera_df.set_index('timestamp')
    motion_df = motion_df.set_index('timestamp')
    
    start_time = max(camera_df.index.min(), motion_df.index.min())
    end_time = min(camera_df.index.max(), motion_df.index.max())
    
    sync_report.append(f"Overlap window: {start_time} to {end_time}")
    
    common_freq = '33ms'
    common_index = pd.date_range(start=start_time, end=end_time, freq=common_freq)
    sync_report.append(f"Created {len(common_index)} synchronized time points at 30Hz")
    
    camera_resampled = camera_df.reindex(common_index, method=method)
    motion_resampled = motion_df.reindex(common_index, method=method)
    
    synchronized = pd.DataFrame(index=common_index)
    synchronized['timestamp'] = synchronized.index
    
    for col in camera_df.columns:
        if col != 'timestamp':
            synchronized[f'camera_{col}'] = camera_resampled[col].values
    
    for col in motion_df.columns:
        if col != 'timestamp':
            synchronized[f'motion_{col}'] = motion_resampled[col].values
    
    if log_df is not None:
        log_df = log_df.copy()
        log_df['timestamp'] = pd.to_datetime(log_df['timestamp'])
        
        for idx, row in log_df.iterrows():
            time_diff = abs(synchronized.index - row['timestamp'])
            nearest_idx = time_diff.argmin()
            if time_diff[nearest_idx] < pd.Timedelta('100ms'):
                col_name = f"event_{row['event_type']}"
                if col_name not in synchronized.columns:
                    synchronized[col_name] = 0
                synchronized.iloc[nearest_idx, synchronized.columns.get_loc(col_name)] = 1
        
        sync_report.append(f"Mapped {len(log_df)} log events to synchronized timeline")
    
    synchronized = synchronized.dropna()
    sync_report.append(f"Final synchronized dataset: {len(synchronized)} samples")
    
    synchronized = synchronized.reset_index(drop=True)
    
    return synchronized, sync_report


tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Input", "🧹 Data Cleaning", "🔄 Synchronization", "📈 Visualization", "💾 Export"])

with tab1:
    st.header("Sensor Data Input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Generate Sample Data")
        st.markdown("Generate realistic multi-sensor data for testing")
        
        if st.button("Generate Sample Dataset", type="primary"):
            start_time = datetime.now()
            st.session_state.camera_data = generate_sample_camera_data(start_time=start_time)
            st.session_state.motion_data = generate_sample_motion_data(start_time=start_time)
            st.session_state.log_data = generate_sample_log_data(start_time=start_time)
            st.success("Sample data generated successfully!")
    
    with col2:
        st.subheader("Upload Your Data")
        st.markdown("Upload CSV files from your sensors")
        
        camera_file = st.file_uploader("Camera Data (CSV)", type=['csv'], key='camera')
        if camera_file:
            st.session_state.camera_data = pd.read_csv(camera_file)
            st.success("Camera data loaded!")
        
        motion_file = st.file_uploader("Motion Sensor Data (CSV)", type=['csv'], key='motion')
        if motion_file:
            st.session_state.motion_data = pd.read_csv(motion_file)
            st.success("Motion data loaded!")
        
        log_file = st.file_uploader("External Logs (CSV)", type=['csv'], key='logs')
        if log_file:
            st.session_state.log_data = pd.read_csv(log_file)
            st.success("Log data loaded!")
    
    st.divider()
    st.subheader("Current Data Status")
    
    status_cols = st.columns(3)
    with status_cols[0]:
        if st.session_state.camera_data is not None:
            st.metric("Camera Data", f"{len(st.session_state.camera_data)} samples")
            with st.expander("Preview Camera Data"):
                st.dataframe(st.session_state.camera_data.head(10), use_container_width=True)
        else:
            st.info("No camera data loaded")
    
    with status_cols[1]:
        if st.session_state.motion_data is not None:
            st.metric("Motion Data", f"{len(st.session_state.motion_data)} samples")
            with st.expander("Preview Motion Data"):
                st.dataframe(st.session_state.motion_data.head(10), use_container_width=True)
        else:
            st.info("No motion data loaded")
    
    with status_cols[2]:
        if st.session_state.log_data is not None:
            st.metric("Log Events", f"{len(st.session_state.log_data)} events")
            with st.expander("Preview Log Data"):
                st.dataframe(st.session_state.log_data.head(10), use_container_width=True)
        else:
            st.info("No log data loaded")

with tab2:
    st.header("Data Cleaning")
    st.markdown("Remove noise, handle missing values, and fix alignment errors")
    
    if st.session_state.camera_data is None and st.session_state.motion_data is None:
        st.warning("Please load data first in the Data Input tab")
    else:
        clean_cols = st.columns(3)
        
        with clean_cols[0]:
            st.subheader("Camera Data")
            if st.session_state.camera_data is not None:
                if st.button("Clean Camera Data"):
                    cleaned, report = clean_data(st.session_state.camera_data, 'camera')
                    st.session_state.camera_data = cleaned
                    for item in report:
                        st.write(f"✓ {item}")
                    st.success(f"Cleaned! {len(cleaned)} samples remaining")
        
        with clean_cols[1]:
            st.subheader("Motion Data")
            if st.session_state.motion_data is not None:
                if st.button("Clean Motion Data"):
                    cleaned, report = clean_data(st.session_state.motion_data, 'motion')
                    st.session_state.motion_data = cleaned
                    for item in report:
                        st.write(f"✓ {item}")
                    st.success(f"Cleaned! {len(cleaned)} samples remaining")
        
        with clean_cols[2]:
            st.subheader("Log Data")
            if st.session_state.log_data is not None:
                if st.button("Clean Log Data"):
                    cleaned, report = clean_data(st.session_state.log_data, 'log')
                    st.session_state.log_data = cleaned
                    for item in report:
                        st.write(f"✓ {item}")
                    st.success(f"Cleaned! {len(cleaned)} samples remaining")

with tab3:
    st.header("Data Synchronization")
    st.markdown("Align data from multiple sensors to a common timeline")
    
    if st.session_state.camera_data is None or st.session_state.motion_data is None:
        st.warning("Please load at least camera and motion data first")
    else:
        st.subheader("Synchronization Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            sync_method = st.selectbox(
                "Interpolation Method",
                ['nearest', 'pad', 'backfill'],
                help="How to align data points between sensors"
            )
        
        if st.button("Synchronize Sensors", type="primary"):
            with st.spinner("Synchronizing sensor data..."):
                synced, report = synchronize_sensors(
                    st.session_state.camera_data,
                    st.session_state.motion_data,
                    st.session_state.log_data,
                    method=sync_method
                )
                st.session_state.synchronized_data = synced
                
                st.subheader("Synchronization Report")
                for item in report:
                    st.write(f"✓ {item}")
                
                if synced is not None:
                    st.success("Synchronization complete!")
                    
                    with st.expander("Preview Synchronized Data"):
                        st.dataframe(synced.head(20), use_container_width=True)

with tab4:
    st.header("Data Visualization")
    
    if st.session_state.synchronized_data is None:
        st.warning("Please synchronize data first")
    else:
        synced = st.session_state.synchronized_data
        
        st.subheader("Multi-Sensor Timeline")
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=('Camera Object Tracking', 'Accelerometer', 'Gyroscope'),
            vertical_spacing=0.08
        )
        
        if 'camera_object_x' in synced.columns:
            fig.add_trace(
                go.Scatter(y=synced['camera_object_x'], name='Object X', line=dict(color='blue')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(y=synced['camera_object_y'], name='Object Y', line=dict(color='red')),
                row=1, col=1
            )
        
        if 'motion_accel_x' in synced.columns:
            fig.add_trace(
                go.Scatter(y=synced['motion_accel_x'], name='Accel X', line=dict(color='green')),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(y=synced['motion_accel_y'], name='Accel Y', line=dict(color='orange')),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(y=synced['motion_accel_z'], name='Accel Z', line=dict(color='purple')),
                row=2, col=1
            )
        
        if 'motion_gyro_x' in synced.columns:
            fig.add_trace(
                go.Scatter(y=synced['motion_gyro_x'], name='Gyro X', line=dict(color='cyan')),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(y=synced['motion_gyro_y'], name='Gyro Y', line=dict(color='magenta')),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(y=synced['motion_gyro_z'], name='Gyro Z', line=dict(color='yellow')),
                row=3, col=1
            )
        
        fig.update_layout(height=700, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        st.subheader("Object Trajectory (Camera)")
        
        if 'camera_object_x' in synced.columns and 'camera_object_y' in synced.columns:
            fig2 = px.scatter(
                synced,
                x='camera_object_x',
                y='camera_object_y',
                color=synced.index,
                title='Object Movement Path',
                labels={'camera_object_x': 'X Position', 'camera_object_y': 'Y Position'}
            )
            fig2.update_layout(height=500)
            st.plotly_chart(fig2, use_container_width=True)
        
        st.divider()
        st.subheader("Correlation Analysis")
        
        numeric_cols = synced.select_dtypes(include=[np.number]).columns.tolist()
        if 'timestamp' in numeric_cols:
            numeric_cols.remove('timestamp')
        
        if len(numeric_cols) > 1:
            corr_matrix = synced[numeric_cols].corr()
            fig3 = px.imshow(
                corr_matrix,
                title='Sensor Correlation Matrix',
                color_continuous_scale='RdBu_r',
                aspect='auto'
            )
            fig3.update_layout(height=600)
            st.plotly_chart(fig3, use_container_width=True)

with tab5:
    st.header("Export Data")
    
    if st.session_state.synchronized_data is None:
        st.warning("Please synchronize data first to export")
    else:
        st.subheader("Download Synchronized Dataset")
        
        csv_buffer = io.StringIO()
        st.session_state.synchronized_data.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="Download as CSV",
            data=csv_data,
            file_name="synchronized_sensor_data.csv",
            mime="text/csv",
            type="primary"
        )
        
        st.divider()
        st.subheader("Dataset Summary")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", len(st.session_state.synchronized_data))
        with col2:
            st.metric("Total Columns", len(st.session_state.synchronized_data.columns))
        with col3:
            st.metric("Memory Size", f"{st.session_state.synchronized_data.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        st.subheader("Column Summary")
        st.dataframe(
            st.session_state.synchronized_data.describe(),
            use_container_width=True
        )

st.sidebar.title("About")
st.sidebar.info("""
**Multi-Sensor Data Pipeline**

This tool helps robotics researchers:
- Record data from multiple sensors
- Clean and preprocess raw data
- Synchronize data to a common timeline
- Visualize and analyze results
- Export for further experiments

**Supported Sensors:**
- Camera (object tracking)
- Motion sensors (IMU)
- External event logs
""")

st.sidebar.divider()
st.sidebar.markdown("Built for Robotics Experiments")
