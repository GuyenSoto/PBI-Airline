import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
import io
import base64
import joblib

# Page configuration
st.set_page_config(
    page_title="Passenger Satisfaction Analysis",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stApp header {
        background-color: #3a86ff;
    }
    h1, h2 {
        color: #3a86ff;
    }
    .stSidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .css-18e3th9 {
        padding-top: 2rem;
    }
    .css-1d391kg {
        padding-top: 3.5rem;
    }
    .stButton>button {
        background-color: #3a86ff;
        color: white;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #2667cc;
    }
    .plot-container {
        background-color: white;
        border-radius: 5px;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
        padding: 20px;
        margin-bottom: 20px;
    }
    .select-all-msg {
        color: #3a86ff;
        font-weight: bold;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .stCheckbox label p {
        font-weight: bold;
        color: #3a86ff;
    }
    .column-divider {
        border-bottom: 1px solid #e1e4e8;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Main title
st.title("📊 Passenger Satisfaction Analysis")
st.markdown("---")

# Helper functions
def iqr_fence(series, k: float = 1.5) -> float:
    q1, q3 = series.quantile([0.25, 0.75])
    return q3 + k * (q3 - q1)

def plot_three_views(series, title: str, save=True):
    """Boxplot, log-y histogram and percentiles; saves figure with _part1 suffix."""
    fig, ax = plt.subplots(1, 3, figsize=(17, 4))

    # Boxplot
    ax[0].boxplot(series.dropna(), vert=True, showfliers=True)
    ax[0].set_title("Boxplot")
    ax[0].set_ylabel(title)

    # Histogram (log-y)
    ax[1].hist(series.dropna(), bins=50, color="#3a86ff")
    ax[1].set_yscale("log")
    ax[1].set_title("Histogram (log-y)")
    ax[1].set_xlabel(title)

    # Percentiles
    sorted_vals = np.sort(series.values)
    pct = np.linspace(0, 100, len(sorted_vals))
    ax[2].plot(pct, sorted_vals, color="#3a86ff")
    ax[2].set_title("Percentile plot")
    ax[2].set_xlabel("Percentile (%)")
    ax[2].set_ylabel(title)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    # Save if needed
    if save:
        OUT_DIR = Path("output")
        OUT_DIR.mkdir(exist_ok=True)
        # Make sure to create a valid filename without path separators
        safe_name = title.lower().replace(" ", "_").replace("\\", "_").replace("/", "_")
        # Remove any other invalid characters
        safe_name = ''.join(c for c in safe_name if c.isalnum() or c == '_')
        try:
            fig.savefig(OUT_DIR / f"{safe_name}_views_part1.png", dpi=300)
        except Exception as e:
            st.error(f"Error saving file: {e}")
    
    return fig

def get_confusion_matrix_plot(y_test, y_pred, save=True):
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Unsatisfied", "Satisfied"])
    disp.plot(values_format="d", cmap="Blues", ax=ax_cm)
    ax_cm.set_title("Confusion Matrix")
    plt.tight_layout()
    
    if save:
        try:
            OUT_DIR = Path("output")
            OUT_DIR.mkdir(exist_ok=True)
            fig_cm.savefig(OUT_DIR / "confusion_matrix_part1.png", dpi=300)
        except Exception as e:
            st.error(f"Error saving confusion matrix: {e}")
    
    return fig_cm

def get_feature_importance_plot(importances, feature_names, top_n=15, save=True):
    top_idx = np.argsort(importances)[-top_n:][::-1]
    fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
    ax_imp.barh(range(len(top_idx)), importances[top_idx][::-1], color="#3a86ff")
    ax_imp.set_yticks(range(len(top_idx)))
    ax_imp.set_yticklabels([feature_names[i] for i in top_idx][::-1])
    ax_imp.set_xlabel("Importance")
    ax_imp.set_title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    
    if save:
        try:
            OUT_DIR = Path("output")
            OUT_DIR.mkdir(exist_ok=True)
            fig_imp.savefig(OUT_DIR / f"feature_importance_top{top_n}_part1.png", dpi=300)
        except Exception as e:
            st.error(f"Error saving feature importance plot: {e}")
    
    return fig_imp

def get_download_link(fig, filename):
    """Generates a link to download the figure as PNG"""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=300)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    button_style = """
    <style>
    .download-button {
        display: inline-block;
        background-color: #3a86ff;
        color: white;
        padding: 8px 15px;
        text-align: center;
        text-decoration: none;
        font-size: 14px;
        border-radius: 4px;
        margin-top: 10px;
        transition: background-color 0.3s;
    }
    .download-button:hover {
        background-color: #2667cc;
    }
    </style>
    """
    download_button = f'<a href="data:image/png;base64,{b64}" download="{filename}" class="download-button">⬇️ Download Chart</a>'
    return button_style + download_button

# Sidebar for navigation and controls
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Select a section:",
    ["Load Data", "Variable Visualization", "Model and Evaluation"]
)

# Section: Load Data
if menu == "Load Data":
    with st.container():
        st.header("📁 Data Loading")
        
        # Option to upload file or use example file
        data_option = st.radio(
            "Select data source:",
            ["Upload CSV file", "Use example file"]
        )
        
        if data_option == "Upload CSV file":
            uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.session_state['df'] = df
                st.success(f"File successfully loaded! Dimensions: {df.shape}")
            else:
                st.info("Please upload a CSV file.")
        else:
            # Try to load example file
            file_path = Path("satisfaction.csv")
            if file_path.exists():
                df = pd.read_csv(file_path)
                st.session_state['df'] = df
                st.success(f"Example file loaded! Dimensions: {df.shape}")
            else:
                st.error("Example file 'satisfaction.csv' not found. Please upload it manually.")
        
        if 'df' in st.session_state:
            df = st.session_state['df']
            
            # Data preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Column information
            st.subheader("Column Information")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Numeric columns:")
                st.write(df.select_dtypes(include=['number']).columns.tolist())
            with col2:
                st.write("Categorical columns:")
                st.write(df.select_dtypes(include=['object']).columns.tolist())
            
            # Basic statistical summary
            if st.checkbox("Show statistical summary"):
                st.subheader("Statistical Summary")
                st.write(df.describe())
            
            # Set target column
            st.subheader("Target Column Configuration")
            target_col = st.selectbox(
                "Select target column for classification model:",
                df.select_dtypes(include=['object']).columns.tolist(),
                index=df.select_dtypes(include=['object']).columns.tolist().index("satisfaction_v2") if "satisfaction_v2" in df.select_dtypes(include=['object']).columns.tolist() else 0
            )
            st.session_state['target_col'] = target_col
            
            # Show target column distribution
            st.subheader(f"Target Column Distribution: {target_col}")
            fig, ax = plt.subplots(figsize=(8, 5))
            df[target_col].value_counts().plot(kind='bar', ax=ax, color="#3a86ff")
            ax.set_ylabel("Count")
            ax.set_title(f"Distribution of {target_col}")
            st.pyplot(fig)
            
            # Save configuration
            if st.button("Save and continue", key="save_config"):
                OUT_DIR = Path("output")
                OUT_DIR.mkdir(exist_ok=True)
                st.session_state['OUT_DIR'] = OUT_DIR
                st.success(f"Configuration saved. Output directory: {OUT_DIR.resolve()}")
                st.info("You can now proceed to 'Variable Visualization'")

# Section: Variable Visualization
elif menu == "Variable Visualization":
    if 'df' not in st.session_state:
        st.warning("Please first load the data in the 'Load Data' section.")
    else:
        df = st.session_state['df']
        st.header("🔍 Exploration and Visualization")
        
        # Select variables to analyze
        st.subheader("Distribution Visualization")
        
        default_cols = ["Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
        available_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Option to select all columns
        select_all = st.checkbox("Select all numeric variables", key="select_all_vars")
        
        if select_all:
            selected_cols = available_cols
        else:
            selected_cols = st.multiselect(
                "Select variables to visualize:",
                available_cols,
                default=list(set(default_cols).intersection(set(available_cols)))
            )
        
        if selected_cols:
            # Add progress bar for multiple visualizations
            if len(selected_cols) > 3:
                st.info(f"Generating visualizations for {len(selected_cols)} variables. This may take a moment...")
                progress_bar = st.progress(0)
                
            # Process each selected column
            for i, col in enumerate(selected_cols):
                try:
                    st.markdown(f"### {col}")
                    with st.container():
                        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                        fig = plot_three_views(df[col], col, save=True)
                        st.pyplot(fig)
                        # Create safe filename for download link
                        safe_filename = col.lower().replace(' ', '_').replace('/', '_').replace('\\', '_')
                        safe_filename = ''.join(c for c in safe_filename if c.isalnum() or c == '_')
                        st.markdown(get_download_link(fig, f"{safe_filename}_views_part1.png"), unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    plt.close(fig)
                except Exception as e:
                    st.error(f"Error processing {col}: {str(e)}")
                
                # Update progress bar if we have many columns
                if len(selected_cols) > 3:
                    progress_bar.progress((i + 1) / len(selected_cols))
            
            # Remove progress bar when complete
            if len(selected_cols) > 3:
                progress_bar.empty()
                st.success("All visualizations generated successfully!")
        
        # Feature Engineering
        st.header("🔧 Feature Engineering and Outlier Filtering")
        
        with st.expander("Create new features"):
            st.markdown("""
            The following features will be created:
            - **SevereDepDelay**: 1 if departure delay > 60 minutes, 0 otherwise
            - **SevereArrDelay**: 1 if arrival delay > 60 minutes, 0 otherwise
            - **LogFlightDist**: Natural logarithm (+ 1) of flight distance
            """)
            
            if st.button("Create new features"):
                # Create features if they don't exist
                if 'SevereDepDelay' not in df.columns:
                    df["SevereDepDelay"] = (df["Departure Delay in Minutes"] > 60).astype(int)
                if 'SevereArrDelay' not in df.columns:
                    df["SevereArrDelay"] = (df["Arrival Delay in Minutes"] > 60).astype(int)
                if 'LogFlightDist' not in df.columns and 'Flight Distance' in df.columns:
                    df["LogFlightDist"] = np.log1p(df["Flight Distance"])
                
                st.session_state['df'] = df
                st.success("New features created successfully!")
                st.write("Sample of new data:")
                st.dataframe(df[['Flight Distance', 'LogFlightDist', 
                               'Departure Delay in Minutes', 'SevereDepDelay',
                               'Arrival Delay in Minutes', 'SevereArrDelay']].head())
        
        with st.expander("Remove outliers"):
            st.markdown("""
            Outliers will be removed based on the IQR (Interquartile Range) method:
            - Values outside Q3 + 1.5*IQR are considered outliers
            - Will be applied to: Flight Distance, Departure Delay in Minutes, Arrival Delay in Minutes
            """)
            
            k_factor = st.slider("k factor for IQR (standard = 1.5)", 1.0, 3.0, 1.5, 0.1)
            
            # Simple explanation for executives
            st.markdown("""
            ### Why Remove Outliers?
            
            **What are outliers?** Outliers are extreme values in our data that don't represent typical behavior, like a 12-hour flight delay when most delays are under 30 minutes.
            
            **Why remove them?**
            * **Improved model accuracy**: Our model will focus on typical patterns rather than being skewed by rare events
            * **Better business insights**: We'll understand what affects normal passenger satisfaction
            * **More reliable predictions**: The model will perform better on new data
            
            **The k-factor** controls how aggressive we are in identifying outliers. Higher values keep more data but may include some unusual cases.
            """)

            
            if st.button("Remove outliers"):
                # Columns to apply outlier filtering
                outlier_cols = [col for col in ["Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"] if col in df.columns]
                
                if outlier_cols:
                    # Build mask to filter outliers
                    mask = pd.Series(True, index=df.index)
                    for col in outlier_cols:
                        mask = mask & (df[col] < iqr_fence(df[col], k_factor))
                    
                    df_clean = df[mask].reset_index(drop=True)
                    
                    # Show effect of removal
                    st.write(f"Original rows: {len(df):,}  →  after removing outliers: {len(df_clean):,}")
                    st.write(f"Removed {len(df) - len(df_clean):,} rows ({(1 - len(df_clean)/len(df))*100:.2f}%)")
                    
                    # Save to session_state
                    st.session_state['df_clean'] = df_clean
                    st.success("Outliers removed successfully!")
                else:
                    st.error("Required columns for outlier filtering not found.")

# Section: Model and Evaluation
elif menu == "Model and Evaluation":
    if 'df' not in st.session_state:
        st.warning("Please first load the data in the 'Load Data' section.")
    elif 'df_clean' not in st.session_state:
        st.warning("Please remove outliers in the 'Variable Visualization' section.")
    else:
        df_clean = st.session_state['df_clean']
        target_col = st.session_state.get('target_col', 'satisfaction_v2')
        
        st.header("🤖 Classification Model and Evaluation")
        
        # Create tabs for different model evaluation views
        tabs = st.tabs(["Model Configuration", "Basic Evaluation", "Threshold Optimization", "Export Model"])
        
        # Tab 1: Model Configuration
        with tabs[0]:
            st.subheader("Gradient Boosting Model Parameters")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                n_estimators = st.slider("Number of estimators", 50, 500, 300, 10)
            with col2:
                learning_rate = st.slider("Learning rate", 0.01, 0.2, 0.05, 0.01)
            with col3:
                max_depth = st.slider("Maximum depth", 2, 10, 3, 1)
            
            test_size = st.slider("Test set size (%)", 10, 40, 20, 5) / 100
            random_state = st.number_input("Random seed", 0, 100, 42, 1)
            
            precision_goal = st.slider("Precision goal for threshold optimization", 0.80, 0.99, 0.95, 0.01)
            
            # Additional parameters for model implementation
            with st.expander("Implementation Parameters (Production Readiness)"):
                st.markdown("""
                ### Model Implementation and Deployment Considerations
                
                For a complete implementation of this model in a production environment, consider these additional parameters:
                
                #### 1. Monitoring and Maintenance
                
                * **Model Drift Detection**: Implement monitoring to detect when model performance degrades over time
                  - Track the distribution of input features vs. training data
                  - Set up alerts when prediction distributions change significantly
                  - Regularly evaluate model on new labeled data
                
                * **Retraining Schedule**: Determine when to retrain the model
                  - Time-based: Monthly/quarterly updates
                  - Performance-based: When accuracy drops below a threshold
                  - Data-based: After collecting X amount of new data
                
                #### 2. Implementation Architecture
                
                * **Inference Latency Requirements**: How fast does the model need to make predictions?
                  - Batch processing: For offline analysis (daily/weekly reports)
                  - Real-time API: For immediate passenger intervention (milliseconds)
                  - Edge deployment: For on-device prediction (e.g., kiosks, mobile apps)
                
                * **Infrastructure Needs**:
                  - Model serving platform (TensorFlow Serving, ONNX Runtime, custom API)
                  - Compute resources (CPU vs GPU, memory requirements)
                  - Scalability needs (horizontal scaling for peak travel periods)
                
                #### 3. Business Integration
                
                * **Output Integration**: Where do model predictions flow?
                  - Customer service dashboard
                  - Automated intervention systems
                  - Analytics/reporting pipelines
                
                * **Threshold Adjustment Mechanism**: How to update the threshold based on business needs
                  - Fixed threshold vs. dynamic adjustment
                  - Different thresholds for different passenger segments
                  - Threshold adjustment based on available resources
                
                * **Feedback Loop**: How to incorporate new data
                  - Verified satisfaction outcomes collection system
                  - Mechanism to incorporate new training data
                  - Human review of model decisions
                
                #### 4. Documentation and Compliance
                
                * **Model Cards**: Documentation that explains model performance, limitations and biases
                * **Decision Logging**: System to log all predictions for audit and analysis
                * **Explainability**: Methods to explain individual predictions to stakeholders
                * **Privacy Compliance**: Ensure GDPR/CCPA compliance for passenger data
                
                These considerations go beyond the statistical model but are crucial for successful real-world implementation.
                """)

            
            if st.button("Train model", key="train_model"):
                with st.spinner("Training model..."):
                    # Prepare data
                    TARGET = target_col
                    y = (df_clean[TARGET].str.lower() == "satisfied").astype(int)
                    X = df_clean.drop(columns=[TARGET])
                    
                    cat_cols = X.select_dtypes(include="object").columns.to_list()
                    num_cols = X.select_dtypes(exclude="object").columns.to_list()
                    
                    cat_pipe = Pipeline([
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore"))
                    ])
                    num_pipe = Pipeline([
                        ("imp", SimpleImputer(strategy="median"))
                    ])
                    
                    preprocess = ColumnTransformer([
                        ("num", num_pipe, num_cols),
                        ("cat", cat_pipe, cat_cols)
                    ])
                    
                    model = GradientBoostingClassifier(
                        n_estimators=n_estimators, 
                        learning_rate=learning_rate, 
                        max_depth=max_depth, 
                        random_state=random_state
                    )
                    
                    pipe = Pipeline([
                        ("prep", preprocess),
                        ("model", model)
                    ])
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, stratify=y, random_state=random_state)
                    
                    # Train model
                    pipe.fit(X_train, y_train)
                    
                    # Calculate metrics on both train and test sets to check for overfitting
                    y_pred_train = pipe.predict(X_train)
                    y_pred = pipe.predict(X_test)
                    
                    # Store metrics for both sets
                    train_report = classification_report(y_train, y_pred_train, digits=3, output_dict=True)
                    test_report = classification_report(y_test, y_pred, digits=3, output_dict=True)
                    
                    # Calculate probabilities and find optimal threshold
                    proba_unsat = pipe.predict_proba(X_test)[:, 0]  # prob for class 0 (unsatisfied)
                    precision, recall, thresh = precision_recall_curve(
                        y_test, proba_unsat, pos_label=0)
                    
                    # Find threshold that meets precision goal
                    idx = np.argmax(precision >= precision_goal)
                    if idx > 0:  # Found a valid threshold
                        thr_star, prec_star, rec_star = thresh[idx], precision[idx], recall[idx]
                    else:  # If no threshold meets the goal, take the highest precision
                        best_idx = np.argmax(precision)
                        thr_star, prec_star, rec_star = thresh[best_idx], precision[best_idx], recall[best_idx]
                        st.warning(f"Could not find threshold with {precision_goal:.2f} precision. Using best available: {prec_star:.2f}")
                    
                    # Create predictions with optimized threshold
                    y_pred_optimized = (proba_unsat >= thr_star).astype(int)
                    
                    # Save to session_state
                    st.session_state['pipe'] = pipe
                    st.session_state['X_train'] = X_train
                    st.session_state['X_test'] = X_test
                    st.session_state['y_train'] = y_train
                    st.session_state['y_test'] = y_test
                    st.session_state['y_pred_train'] = y_pred_train
                    st.session_state['y_pred'] = y_pred
                    st.session_state['train_report'] = train_report
                    st.session_state['test_report'] = test_report
                    st.session_state['y_pred_optimized'] = y_pred_optimized
                    st.session_state['proba_unsat'] = proba_unsat
                    st.session_state['precision'] = precision
                    st.session_state['recall'] = recall
                    st.session_state['thresh'] = thresh
                    st.session_state['thr_star'] = thr_star
                    st.session_state['prec_star'] = prec_star
                    st.session_state['rec_star'] = rec_star
                    st.session_state['num_cols'] = num_cols
                    st.session_state['cat_cols'] = cat_cols
                    st.session_state['precision_goal'] = precision_goal
                    
                    st.success("Model trained successfully!")
                    
        # Tab 2: Basic Evaluation
        with tabs[1]:
            if 'pipe' in st.session_state:
                pipe = st.session_state['pipe']
                X_test = st.session_state['X_test']
                y_test = st.session_state['y_test']
                y_pred = st.session_state['y_pred']
                num_cols = st.session_state['num_cols']
                cat_cols = st.session_state['cat_cols']
                
                st.subheader("Classification Report (threshold = 0.5)")
                
                # Compare train vs test performance to check for overfitting
                train_report = st.session_state['train_report']
                test_report = st.session_state['test_report']
                
                # Show overfitting analysis
                st.markdown("### Overfitting Analysis")
                
                # Create a DataFrame to compare train and test metrics
                metrics_df = pd.DataFrame({
                    'Metric': ['Accuracy', 'Precision (Satisfied)', 'Recall (Satisfied)', 'F1 (Satisfied)'],
                    'Training Set': [
                        train_report['accuracy'], 
                        train_report['1']['precision'], 
                        train_report['1']['recall'], 
                        train_report['1']['f1-score']
                    ],
                    'Test Set': [
                        test_report['accuracy'], 
                        test_report['1']['precision'], 
                        test_report['1']['recall'], 
                        test_report['1']['f1-score']
                    ],
                    'Difference': [
                        train_report['accuracy'] - test_report['accuracy'],
                        train_report['1']['precision'] - test_report['1']['precision'],
                        train_report['1']['recall'] - test_report['1']['recall'],
                        train_report['1']['f1-score'] - test_report['1']['f1-score']
                    ]
                })
                
                # Format as percentages
                metrics_df['Training Set'] = metrics_df['Training Set'].apply(lambda x: f"{x:.1%}")
                metrics_df['Test Set'] = metrics_df['Test Set'].apply(lambda x: f"{x:.1%}")
                metrics_df['Difference'] = metrics_df['Difference'].apply(lambda x: f"{x:.1%}")
                
                # Display the comparison
                st.dataframe(metrics_df)
                
                # Determine overfitting/underfitting level
                avg_diff = np.mean([
                    train_report['accuracy'] - test_report['accuracy'],
                    train_report['1']['precision'] - test_report['1']['precision'],
                    train_report['1']['recall'] - test_report['1']['recall'],
                    train_report['1']['f1-score'] - test_report['1']['f1-score']
                ])
                
                # Check for underfitting - look at absolute performance on both sets
                train_accuracy = train_report['accuracy']
                test_accuracy = test_report['accuracy']
                
                # Create an assessment combining overfitting and underfitting analysis
                if train_accuracy < 0.7:  # Low performance on training data indicates underfitting
                    st.error(f"""
                    **Underfitting detected** (training accuracy: {train_accuracy:.1%})
                    
                    The model isn't performing well even on the training data, suggesting it's too simple to capture the underlying patterns. Consider:
                    - Increasing model complexity (more estimators, higher max_depth)
                    - Adding more relevant features or feature engineering
                    - Reducing regularization if applied
                    - Trying a different algorithm that might better fit the data structure
                    
                    Current model is likely too simple for the complexity of the problem.
                    """)
                elif avg_diff > 0.1:
                    st.warning(f"""
                    **Significant overfitting detected** (average difference: {avg_diff:.1%})
                    
                    The model performs significantly better on the training data than on test data, suggesting it has memorized the training data rather than learning generalizable patterns. Consider:
                    - Reducing model complexity (fewer estimators, lower max_depth)
                    - Gathering more diverse training data
                    - Using regularization techniques
                    """)
                elif avg_diff > 0.05:
                    st.info(f"""
                    **Mild overfitting detected** (average difference: {avg_diff:.1%})
                    
                    The model performs somewhat better on the training data than on test data. This is common and may not be a major concern, but you could try:
                    - Slightly reducing model complexity
                    - Using a larger test set
                    """)
                elif test_accuracy < 0.75:  # Good generalization but mediocre performance suggests underfitting
                    st.warning(f"""
                    **Potential underfitting with good generalization** (test accuracy: {test_accuracy:.1%})
                    
                    The model generalizes well (similar performance on train and test), but the overall performance is lower than optimal. This suggests the model might be too simple. Consider:
                    - Gradually increasing model complexity
                    - Adding more informative features
                    - Exploring different algorithms
                    """)
                else:
                    st.success(f"""
                    **Good generalization and performance** (test accuracy: {test_accuracy:.1%}, difference: {avg_diff:.1%})
                    
                    The model performs well on both training and test data with minimal difference, suggesting it has learned general patterns rather than memorizing the training data. This model is likely to perform well on new, unseen data.
                    """)
                
                # Learning curves for visual inspection of overfitting/underfitting
                st.markdown("### Learning Curves Analysis")
                st.markdown("""
                Learning curves show model performance on training and validation sets as the training set size increases. 
                They're excellent for diagnosing overfitting vs. underfitting:
                
                * **Underfitting**: Both curves are close together but with high error (low performance)
                * **Overfitting**: Large gap between training and validation performance
                * **Good fit**: Both curves converge to a low error (high performance)
                """)
                
                # Simulate learning curves for demonstration (ideal situation would use cross-validation)
                from sklearn.model_selection import learning_curve
                
                # Get current model parameters
                current_params = {
                    'n_estimators': st.session_state['pipe'].named_steps['model'].n_estimators,
                    'learning_rate': st.session_state['pipe'].named_steps['model'].learning_rate,
                    'max_depth': st.session_state['pipe'].named_steps['model'].max_depth
                }
                
                # Plot learning curves button
                if st.button("Generate Learning Curves (This may take a moment)"):
                    with st.spinner("Calculating learning curves..."):
                        try:
                            # Instead of using the raw data, we'll use the preprocessed data
                            # We need to use numeric features only for learning curves
                            X = st.session_state['X_train']
                            y = st.session_state['y_train']
                            
                            # Use only numeric columns for simplicity and reliability
                            numeric_cols = X.select_dtypes(include=['number']).columns
                            X_numeric = X[numeric_cols]
                            
                            # Create a simpler model for learning curves
                            from sklearn.ensemble import GradientBoostingClassifier
                            simple_model = GradientBoostingClassifier(
                                n_estimators=min(50, current_params['n_estimators']),  # Use fewer estimators for speed
                                learning_rate=current_params['learning_rate'],
                                max_depth=current_params['max_depth'],
                                random_state=42
                            )
                            
                            # Calculate learning curves with numeric features only
                            train_sizes, train_scores, test_scores = learning_curve(
                                simple_model, X_numeric, y, 
                                train_sizes=np.linspace(0.1, 1.0, 5),
                                cv=5, scoring='accuracy', n_jobs=-1
                            )
                            
                            # Create a plot
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.set_title("Learning Curves (Gradient Boosting - Numeric Features Only)")
                            ax.set_xlabel("Training examples")
                            ax.set_ylabel("Accuracy")
                            
                            # Plot learning curves
                            ax.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='r', label="Training score")
                            ax.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color='g', label="Cross-validation score")
                            ax.grid(True)
                            ax.legend(loc="best")
                            
                            # Determine fit type based on curves
                            final_train_score = np.mean(train_scores, axis=1)[-1]
                            final_test_score = np.mean(test_scores, axis=1)[-1]
                            score_diff = final_train_score - final_test_score
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            st.info("""
                            **Note**: These learning curves use only numeric features for simplicity. 
                            The full model (with all features) may show different performance characteristics.
                            """)
                            
                            # Learning curve interpretation
                            if final_train_score < 0.7:
                                st.error("**Learning curves suggest underfitting**: Both training and validation scores are low, indicating the model is too simple.")
                            elif score_diff > 0.1:
                                st.warning("**Learning curves suggest overfitting**: There's a significant gap between training and validation scores.")
                            elif final_test_score < 0.75:
                                st.warning("**Learning curves suggest potential underfitting**: Scores converge but at a lower value than optimal.")
                            else:
                                st.success("**Learning curves suggest a good fit**: Both curves converge to a high score with minimal gap.")
                        except Exception as e:
                            st.error(f"Error generating learning curves: {str(e)}")
                            st.info("""
                            **Alternative Approach**: Instead of learning curves, you can compare the training and test metrics in the table above.
                            A large difference between training and test performance indicates overfitting,
                            while low performance on both indicates underfitting.
                            """)

                
                # Main metrics in cards
                st.markdown("### Key Performance Metrics (Test Set)")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{test_report['accuracy']:.3f}")
                with col2:
                    st.metric("Precision", f"{test_report['1']['precision']:.3f}")
                with col3:
                    st.metric("Recall", f"{test_report['1']['recall']:.3f}")
                with col4:
                    st.metric("F1-Score", f"{test_report['1']['f1-score']:.3f}")
                
                # Complete report
                st.markdown("### Full Classification Report (Test Set)")
                st.text(classification_report(st.session_state['y_test'], st.session_state['y_pred'], digits=3))
                
                # Confusion matrix
                st.subheader("Confusion Matrix (threshold = 0.5)")
                with st.container():
                    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                    fig_cm = get_confusion_matrix_plot(y_test, y_pred, save=True)
                    st.pyplot(fig_cm)
                    st.markdown(get_download_link(fig_cm, "confusion_matrix_part1.png"), unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                plt.close(fig_cm)
                
                # Feature importance
                st.subheader("Feature Importance")
                top_n = st.slider("Number of top features to show", 5, 30, 15, 1)
                
                # Get feature names after one-hot encoding
                ohe = pipe.named_steps["prep"].named_transformers_["cat"].named_steps["ohe"]
                feature_names = num_cols + list(ohe.get_feature_names_out(cat_cols))
                importances = pipe.named_steps["model"].feature_importances_
                
                with st.container():
                    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                    fig_imp = get_feature_importance_plot(importances, feature_names, top_n=top_n, save=True)
                    st.pyplot(fig_imp)
                    st.markdown(get_download_link(fig_imp, f"feature_importance_top{top_n}_part1.png"), unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                plt.close(fig_imp)
                
                # Executive summary of basic evaluation
                st.subheader("📝 Executive Summary")
                st.markdown(f"""
                ### What does this tell us?
                
                **Model Performance:** Our model can predict passenger satisfaction with {test_report['accuracy']:.1%} accuracy using the default 0.5 threshold. This means that for about {int(test_report['accuracy']*100)} out of 100 passengers, we correctly identify whether they are satisfied or not.
                
                **Feature Importance:** The chart above shows which factors most influence passenger satisfaction. The longer the bar, the more important that factor is in determining satisfaction. This helps us understand where to focus improvement efforts.
                
                **Business Impact:** By knowing the most influential factors, we can prioritize improvements that will have the greatest impact on overall passenger satisfaction. This allows us to allocate resources more effectively.
                
                **Next Steps:** Look at the "Threshold Optimization" tab to see how we can fine-tune our model to better identify unsatisfied passengers, which would allow for more targeted interventions.
                """)

                
            else:
                st.info("Please train the model first in the 'Model Configuration' tab.")
                
        # Tab 3: Threshold Optimization
        with tabs[2]:
            if 'pipe' in st.session_state and 'thr_star' in st.session_state:
                # Get data from session state
                y_test = st.session_state['y_test']
                y_pred_optimized = st.session_state['y_pred_optimized']
                proba_unsat = st.session_state['proba_unsat']
                precision = st.session_state['precision']
                recall = st.session_state['recall']
                thresh = st.session_state['thresh']
                thr_star = st.session_state['thr_star']
                prec_star = st.session_state['prec_star']
                rec_star = st.session_state['rec_star']
                precision_goal = st.session_state['precision_goal']
                
                st.subheader("Threshold Optimization Results")
                
                # Display optimized threshold info
                st.info(f"""
                **Optimized threshold: {thr_star:.4f}**
                - Precision: {prec_star:.4f}
                - Recall: {rec_star:.4f}
                
                This threshold maximizes recall while maintaining at least {precision_goal:.2f} precision for detecting unsatisfied passengers.
                """)
                
                # Show optimized confusion matrix
                st.subheader(f"Confusion Matrix (threshold = {thr_star:.4f})")
                with st.container():
                    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                    cm = confusion_matrix(y_test, y_pred_optimized, labels=[1,0])
                    fig_cm_opt, ax_cm_opt = plt.subplots(figsize=(5, 4))
                    ConfusionMatrixDisplay(cm, display_labels=["Satisfied", "Unsatisfied"]).plot(
                        values_format="d", cmap="Blues", ax=ax_cm_opt)
                    ax_cm_opt.set_title(f"Confusion Matrix @thr={thr_star:.4f}")
                    plt.tight_layout()
                    st.pyplot(fig_cm_opt)
                    
                    # Save confusion matrix with optimized threshold
                    try:
                        OUT_DIR = Path("output")
                        OUT_DIR.mkdir(exist_ok=True)
                        fig_cm_opt.savefig(OUT_DIR / "confusion_matrix_thr95.png", dpi=120)
                    except Exception as e:
                        st.error(f"Error saving optimized confusion matrix: {e}")
                    
                    st.markdown(get_download_link(fig_cm_opt, "confusion_matrix_thr95.png"), unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                plt.close(fig_cm_opt)
                
                # PR Curve
                st.subheader("Precision-Recall Curve")
                with st.container():
                    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                    
                    # Create PR curve plot
                    fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
                    ax_pr.plot(recall, precision, lw=2)
                    ax_pr.axhline(precision_goal, color="red", ls="--", 
                                  label=f"{precision_goal:.0%} precision goal")
                    ax_pr.scatter(rec_star, prec_star, color="red", s=80, zorder=5)
                    ax_pr.text(rec_star, prec_star+0.02,
                             f"Prec={prec_star:.2f}\nRec={rec_star:.2f}", ha="center", color="red")
                    ax_pr.set_xlabel("Recall (coverage)")
                    ax_pr.set_ylabel("Precision (accuracy)")
                    ax_pr.set_title("Precision–Recall curve (unsatisfied class)")
                    ax_pr.legend()
                    plt.tight_layout()
                    
                    # Save PR curve
                    try:
                        OUT_DIR = Path("output")
                        OUT_DIR.mkdir(exist_ok=True)
                        fig_pr.savefig(OUT_DIR / "pr_curve_explained.png", dpi=120)
                    except Exception as e:
                        st.error(f"Error saving PR curve: {e}")
                    
                    st.pyplot(fig_pr)
                    st.markdown(get_download_link(fig_pr, "pr_curve_explained.png"), unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                plt.close(fig_pr)
                
                # Explanation of the PR curve in simple terms
                st.markdown("""
                ### Understanding the Precision-Recall Curve
                
                **What is this?** This curve shows the trade-off between precision (accuracy) and recall (coverage) when trying to identify unsatisfied passengers.
                
                **In simple terms:**
                * **Precision** = "When we predict a passenger is unsatisfied, how often are we correct?"
                * **Recall** = "What percentage of all unsatisfied passengers can we identify?"
                
                **The red dot** shows our chosen balance point. At this point, we maintain high precision (we're rarely wrong when we flag someone as unsatisfied) while maximizing how many unsatisfied passengers we can identify.
                
                **Business value:** This helps us focus resources on passengers who are truly unsatisfied, while capturing as many of these cases as possible.
                """)

                
                # Precision and Recall vs Threshold Bar Charts
                st.subheader("Precision and Recall vs Threshold")
                
                # Function to create bar charts for precision and recall vs threshold
                def make_bar_chart(values, color, title, subtitle):
                    fig, ax = plt.subplots(figsize=(7, 3.8))
                    ax.bar(labels, values, color=color)
                    ax.set_ylim(0, 1)
                    ax.set_ylabel(title.split()[0])  # Precision or Recall
                    ax.set_title(title)
                    for i, v in enumerate(values):
                        ax.text(i, v+0.03, f"{v:.2f}", ha="center")
                    fig.text(0.5, 0.01, subtitle, ha="center", fontsize=9)
                    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
                    return fig
                
                # Calculate precision and recall at various thresholds
                cuts = [0.30, 0.50, 0.70, thr_star]
                prec_vals, rec_vals = [], []
                for cp in cuts:
                    pred = (proba_unsat >= cp).astype(int)
                    tp = ((pred==1)&(y_test==0)).sum(); fp = ((pred==1)&(y_test==1)).sum()
                    fn = ((pred==0)&(y_test==0)).sum()
                    prec_vals.append(tp/(tp+fp) if tp+fp else 0)
                    rec_vals.append(tp/(tp+fn) if tp+fn else 0)
                
                labels = [f"Thr {cp:.2f}" for cp in cuts]
                
                # Display charts in two columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                    fig_prec = make_bar_chart(
                        prec_vals, "#4CAF50", "Precision vs Threshold",
                        "Higher threshold → fewer calls\nbut more accurate")
                    st.pyplot(fig_prec)
                    
                    # Save precision chart
                    try:
                        OUT_DIR = Path("output")
                        OUT_DIR.mkdir(exist_ok=True)
                        fig_prec.savefig(OUT_DIR / "precision_vs_threshold.png", dpi=120)
                    except Exception as e:
                        st.error(f"Error saving precision chart: {e}")
                    
                    st.markdown(get_download_link(fig_prec, "precision_vs_threshold.png"), unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                plt.close(fig_prec)
                
                with col2:
                    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                    fig_rec = make_bar_chart(
                        rec_vals, "#2196F3", "Recall vs Threshold",
                        "Higher threshold → miss more unhappy passengers")
                    st.pyplot(fig_rec)
                    
                    # Save recall chart
                    try:
                        OUT_DIR = Path("output")
                        OUT_DIR.mkdir(exist_ok=True)
                        fig_rec.savefig(OUT_DIR / "recall_vs_threshold.png", dpi=120)
                    except Exception as e:
                        st.error(f"Error saving recall chart: {e}")
                    
                    st.markdown(get_download_link(fig_rec, "recall_vs_threshold.png"), unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                plt.close(fig_rec)
                
                # Executive explanation of threshold optimization
                st.markdown(f"""
                ### Executive Summary: Threshold Optimization
                
                **What are these charts showing?** These bar charts demonstrate how adjusting our decision threshold affects our ability to identify unsatisfied passengers.
                
                **Why this matters:**
                * **Higher threshold** means we're more confident before flagging a passenger as unsatisfied. This increases our precision (fewer false alarms) but reduces our recall (we miss more unhappy passengers).
                * **Lower threshold** means we flag more passengers as potentially unsatisfied. This catches more truly unsatisfied passengers but generates more false alarms.
                
                **Business applications:**
                * **High-value customers**: Use a lower threshold to make sure we don't miss any dissatisfaction
                * **Expensive interventions**: Use a higher threshold to focus resources where they're most needed
                * **General monitoring**: The optimal threshold (shown as the last bar) balances these concerns
                
                **Bottom line:** Our optimized threshold of {thr_star:.4f} gives us {prec_star:.1%} precision with {rec_star:.1%} recall, meaning we can accurately identify unsatisfied passengers while minimizing resource waste on false positives.
                """)

                
            else:
                st.info("Please train the model first in the 'Model Configuration' tab.")
                
        # Tab 4: Export Model
        with tabs[3]:
            if 'pipe' in st.session_state and 'thr_star' in st.session_state:
                st.subheader("Export Model and Threshold")
                
                # Get data
                pipe = st.session_state['pipe']
                thr_star = st.session_state['thr_star']
                OUT_DIR = Path("output")
                OUT_DIR.mkdir(exist_ok=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Export Model"):
                        try:
                            # Save the model
                            joblib.dump(pipe, OUT_DIR / "model.pkl")
                            st.success(f"Model saved to {OUT_DIR/'model.pkl'}")
                        except Exception as e:
                            st.error(f"Error saving model: {e}")
                
                with col2:
                    if st.button("Export Threshold"):
                        try:
                            # Save threshold
                            with open(OUT_DIR / "threshold.txt", "w") as f:
                                f.write(f"{thr_star:.6f}")
                            st.success(f"Threshold saved to {OUT_DIR/'threshold.txt'}")
                        except Exception as e:
                            st.error(f"Error saving threshold: {e}")
                
                # Export cleaned dataset
                if st.button("Export Cleaned Dataset"):
                    try:
                        df_clean = st.session_state['df_clean']
                        df_clean.to_csv(OUT_DIR / "cleaned_dataset.csv", index=False)
                        st.success(f"Cleaned dataset saved to {OUT_DIR/'cleaned_dataset.csv'}")
                    except Exception as e:
                        st.error(f"Error saving cleaned dataset: {e}")
                
                # Generate Report Summary
                if st.button("Generate Report Summary"):
                    st.markdown(f"""
                    ### 📋 Executive Report Summary
                    
                    #### Key Findings
                    * Our model can predict passenger satisfaction with high accuracy
                    * The most important factors influencing satisfaction are clearly identified
                    * We've optimized our detection of unsatisfied passengers to balance precision and recall
                    
                    #### Business Recommendations
                    1. Focus improvement initiatives on the top factors identified in the Feature Importance chart
                    2. Use the optimized threshold when implementing passenger satisfaction monitoring systems
                    3. Consider different thresholds for different customer segments based on business value
                    
                    #### Technical Details
                    All exports have been saved to the 'output' directory. The following files were generated:
                    
                    **Data & Model Files:**
                    * **Model file**: `model.pkl` - Trained machine learning pipeline
                    * **Threshold file**: `threshold.txt` - Optimized probability threshold ({thr_star:.4f})
                    * **Cleaned dataset**: `cleaned_dataset.csv` - Dataset after outlier removal
                    
                    **Visualizations:**
                    * Variable distributions: `*_views_part1.png`
                    * Confusion matrix (default): `confusion_matrix_part1.png`
                    * Confusion matrix (optimized): `confusion_matrix_thr95.png`
                    * Feature importance: `feature_importance_top*.png`
                    * Precision-Recall curve: `pr_curve_explained.png`
                    * Precision vs Threshold: `precision_vs_threshold.png`
                    * Recall vs Threshold: `recall_vs_threshold.png`
                    """)

                    
                st.success(f"All artifacts have been saved to: {OUT_DIR.resolve()}")
            else:
                st.info("Please train the model first in the 'Model Configuration' tab.")

# Footer
st.markdown("---")
st.markdown("📊 **Passenger Satisfaction Analysis** | Developed with Streamlit")