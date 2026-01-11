
!pip install pyspark -q
!pip install streamlit -q
!pip install pyngrok -q

from google.colab import drive
drive.mount('/content/drive')

import os

BASE = "/content/drive/MyDrive/cloud-ml-service"
for folder in ["uploads", "results", "jobs"]:
    os.makedirs(os.path.join(BASE, folder), exist_ok=True)

print("Ready:",BASE)


%%writefile app.py
import streamlit as st
import pandas as pd
import time
import os
import json
import uuid
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.functions import array, col, lit, concat, when, sum as Fsum
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor
from pyspark.ml.fpm import FPGrowth

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

st.set_page_config(page_title="Cloud ", layout="wide")


DRIVE_BASE = "/content/drive/MyDrive/cloud-ml-service"
UPLOAD_DIR = os.path.join(DRIVE_BASE, "uploads")
RESULTS_DIR = os.path.join(DRIVE_BASE, "results")
JOBS_DIR = os.path.join(DRIVE_BASE, "jobs")

for p in [UPLOAD_DIR, RESULTS_DIR, JOBS_DIR]:
    os.makedirs(p, exist_ok=True)

def new_job():
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    return job_id, job_dir

def write_status(job_dir, status, extra=None):
    payload = {"status": status, "timestamp": datetime.now().isoformat()}
    if extra:
        payload.update(extra)
    with open(os.path.join(job_dir, "status.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

if 'baseline_times' not in st.session_state:
    st.session_state['baseline_times'] = {}

def calculate_metrics(job_name, nodes, duration):
    if nodes == 1:
        st.session_state['baseline_times'][job_name] = duration
        speedup = 1.0
        efficiency = 1.0
    else:
        t1 = st.session_state['baseline_times'].get(job_name)
        if t1 and duration > 0:
            speedup = t1 / duration
            efficiency = speedup / nodes
        else:
            speedup = 0.0
            efficiency = 0.0
    return speedup, efficiency


st.title(" Cloud and Distributed Systems")
uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "json", "txt", "pdf"])

if uploaded_file is not None:
    job_id, job_dir = new_job()
    write_status(job_dir, "RECEIVED")

    file_name = uploaded_file.name
    file_extension = file_name.split('.')[-1].lower()

    safe_name = file_name.replace("/", "").replace("\\", "")
    stored_path = os.path.join(UPLOAD_DIR, f"{job_id}_{safe_name}")

    with open(stored_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    write_status(job_dir, "STORED", {"file_path": stored_path, "file_type": file_extension})

    st.success(f" {file_extension.upper()} File uploaded & stored in Drive!")
    st.info(f" Job ID: {job_id}")
    st.caption(f"Drive path: {stored_path}")

    st.sidebar.header("Processing Options")
    num_nodes = st.sidebar.select_slider("Select Number of  Nodes", options=[1, 2, 4, 8])


    spark = SparkSession.builder \
        .appName("EngineeringProject") \
        .master(f"local[{num_nodes}]") \
        .getOrCreate()

    df = None
    try:
        write_status(job_dir, "VALIDATING")
        if file_extension == 'csv':
            df = spark.read.csv(stored_path, header=True, inferSchema=True)
        elif file_extension == 'json':
            df = spark.read.option("multiline", "true").json(stored_path)
        elif file_extension == 'txt':
            df = spark.read.csv(stored_path, header=True, inferSchema=True, sep="\t")
        elif file_extension == 'pdf':
            if PdfReader:
                reader = PdfReader(stored_path)
                text_content = []
                for page in reader.pages:
                    text_content.append((page.extract_text(),))
                df = spark.createDataFrame(text_content, ["pdf_content"])
                st.warning(" PDF is text-only.")
            else:
                st.error("Please install pypdf: !pip install pypdf")
                write_status(job_dir, "FAILED", {"error": "pypdf not installed"})
        write_status(job_dir, "READY")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        write_status(job_dir, "FAILED", {"error": str(e)})

    if df is not None:
        numeric_cols = [c for c, t in df.dtypes if t in ('int', 'double', 'float', 'bigint', 'long')]
        all_cols = df.columns

        job_type = st.selectbox(
            "2. Select Processing Job",
            [
                "Descriptive Statistics",
                "K-Means Clustering",
                "Linear Regression",
                "Decision Tree (Regression)",
                "FPGrowth "
            ]
        )

        def save_pandas(df_pd, filename):
            path = os.path.join(RESULTS_DIR, f"{job_id}_{filename}")
            df_pd.to_csv(path, index=False)
            return path

        def save_json(obj, filename):
            path = os.path.join(RESULTS_DIR, f"{job_id}_{filename}")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
            return path

        if job_type == "Descriptive Statistics":
            if st.button("Run Analysis"):
                write_status(job_dir, "RUNNING", {"job": "Descriptive Statistics"})
                start_time = time.time()

                rows = df.count()
                cols = len(df.columns)
                dtypes_list = [{"column": c, "dtype": t} for c, t in df.dtypes]

                null_exprs = [
                    (Fsum(when(col(c).isNull(), 1).otherwise(0)) / lit(rows) * 100).alias(c)
                    for c in df.columns
                ]
                null_pct = df.select(*null_exprs).first().asDict()

                uniq_exprs = [pd.Series]


                if not numeric_cols:
                    stats_pd = pd.DataFrame({"rows": [rows], "columns": [cols]})
                else:
                    stats_pd = df.describe().toPandas()

                end_time = time.time()
                duration = end_time - start_time
                sp, eff = calculate_metrics("Statistics", num_nodes, duration)


                stats_csv_path = save_pandas(stats_pd, "stats_table.csv")
                stats_json_path = save_json(
                    {
                        "rows": rows,
                        "columns": cols,
                        "dtypes": dtypes_list,
                        "null_percentage": null_pct,
                        "execution_time_sec": duration,
                        "nodes_setting": num_nodes,
                        "speedup": sp,
                        "efficiency": eff,
                    },
                    "stats_summary.json"
                )

                write_status(job_dir, "FINISHED", {"stats_csv": stats_csv_path, "stats_json": stats_json_path})

                st.subheader(" Results Table")
                st.table(stats_pd)
                st.write(f"Execution Time: {duration:.4f} s | Speedup: {sp:.2f}x | Efficiency: {eff:.2f}")
                st.success("Saved to Drive ")
                st.caption(f"{stats_csv_path}")
                st.caption(f"{stats_json_path}")


        elif job_type == "K-Means Clustering":
            if not numeric_cols:
                st.error("Requires numeric columns.")
            else:
                selected_cols = st.multiselect(
                    "Select Columns",
                    numeric_cols,
                    default=numeric_cols[:2] if len(numeric_cols) > 1 else numeric_cols
                )
                k_val = st.slider("K Clusters", 2, 10, 3)

                if st.button("Run Clustering") and len(selected_cols) >= 2:
                    write_status(job_dir, "RUNNING", {"job": "K-Means", "k": k_val, "cols": selected_cols})
                    start_time = time.time()

                    assembler = VectorAssembler(inputCols=selected_cols, outputCol="features")
                    data = assembler.transform(df.na.drop())
                    kmeans = KMeans().setK(k_val).setSeed(1)
                    model = kmeans.fit(data)

                    end_time = time.time()
                    duration = end_time - start_time
                    sp, eff = calculate_metrics("K-Means", num_nodes, duration)

                    centers = model.clusterCenters()
                    center_df = pd.DataFrame(centers, columns=selected_cols)
                    center_df.index.name = "Cluster ID"
                    center_df_reset = center_df.reset_index()

                    centers_path = save_pandas(center_df_reset, "kmeans_centers.csv")
                    meta_path = save_json(
                        {
                            "k": k_val,
                            "columns": selected_cols,
                            "execution_time_sec": duration,
                            "nodes_setting": num_nodes,
                            "speedup": sp,
                            "efficiency": eff,
                        },
                        "kmeans_meta.json"
                    )

                    write_status(job_dir, "FINISHED", {"centers_csv": centers_path, "meta_json": meta_path})

                    st.success("Done.")
                    st.subheader(" Cluster Centers (Table)")
                    st.table(center_df)
                    st.write(f"Execution Time: {duration:.4f} s | Speedup: {sp:.2f}x | Efficiency: {eff:.2f}")
                    st.success("Saved to Drive ")
                    st.caption(f"{centers_path}")
                    st.caption(f"{meta_path}")


        elif job_type == "Linear Regression":
            if not numeric_cols:
                st.error("Requires numeric columns.")
            else:
                target_col = st.selectbox("Target (Y)", numeric_cols)
                feature_cols = st.multiselect("Features (X)", [c for c in numeric_cols if c != target_col])

                if st.button("Train Model") and feature_cols:
                    write_status(job_dir, "RUNNING", {"job": "Linear Regression", "y": target_col, "x": feature_cols})
                    start_time = time.time()

                    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
                    data = assembler.transform(df.na.drop())

                    lr = LinearRegression(labelCol=target_col)
                    model = lr.fit(data)

                    end_time = time.time()
                    duration = end_time - start_time
                    sp, eff = calculate_metrics("Linear Regression", num_nodes, duration)

                    coefficients = model.coefficients.toArray()
                    coeff_df = pd.DataFrame({"Feature": feature_cols, "Coefficient": coefficients})

                    coeff_path = save_pandas(coeff_df, "linear_regression_coefficients.csv")
                    meta_path = save_json(
                        {
                            "intercept": float(model.intercept),
                            "execution_time_sec": duration,
                            "nodes_setting": num_nodes,
                            "speedup": sp,
                            "efficiency": eff,
                        },
                        "linear_regression_meta.json"
                    )

                    write_status(job_dir, "FINISHED", {"coeff_csv": coeff_path, "meta_json": meta_path})

                    st.success("Model Trained Successfully.")
                    st.subheader("Model Coefficients")
                    st.table(coeff_df)
                    st.write(f"Model Intercept: {model.intercept:.4f}")
                    st.write(f"Execution Time: {duration:.4f} s | Speedup: {sp:.2f}x | Efficiency: {eff:.2f}")
                    st.success("Saved to Drive ")
                    st.caption(f"{coeff_path}")
                    st.caption(f"{meta_path}")


        elif job_type == "Decision Tree (Regression)":
            if not numeric_cols:
                st.error("Requires numeric columns")
            else:
                target_col = st.selectbox("Target (Y)", numeric_cols, key="dt_target")
                feature_cols = st.multiselect("Features (X)", [c for c in numeric_cols if c != target_col], key="dt_features")
                max_depth = st.slider("Max Tree Depth", 2, 5, 3)

                if st.button("Train Decision Tree") and feature_cols:
                    write_status(job_dir, "RUNNING", {"job": "Decision Tree", "y": target_col, "x": feature_cols, "max_depth": max_depth})
                    start_time = time.time()

                    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
                    data = assembler.transform(df.na.drop())
                    train, test = data.randomSplit([0.8, 0.2])

                    dt = DecisionTreeRegressor(labelCol=target_col, maxDepth=max_depth)
                    model = dt.fit(train)

                    predictions = model.transform(test)

                    end_time = time.time()
                    duration = end_time - start_time
                    sp, eff = calculate_metrics("Decision Tree", num_nodes, duration)

                    display_df = predictions.select(feature_cols + ['prediction']).limit(15).toPandas()
                    display_df.rename(columns={'prediction': f'Predicted {target_col}'}, inplace=True)

                    pred_path = save_pandas(display_df, "decision_tree_predictions_sample.csv")
                    meta_path = save_json(
                        {
                            "max_depth": max_depth,
                            "execution_time_sec": duration,
                            "nodes_setting": num_nodes,
                            "speedup": sp,
                            "efficiency": eff,
                        },
                        "decision_tree_meta.json"
                    )

                    write_status(job_dir, "FINISHED", {"pred_sample_csv": pred_path, "meta_json": meta_path})

                    st.success("Model Trained Successfully.")
                    st.subheader("Prediction Results")
                    st.table(display_df)
                    st.write(f"Execution Time: {duration:.4f} s | Speedup: {sp:.2f}x | Efficiency: {eff:.2f}")
                    st.success("Saved to Drive ")
                    st.caption(f"{pred_path}")
                    st.caption(f"{meta_path}")


        elif job_type == "FPGrowth ":
            selected_cols = st.multiselect("Select Categorical Columns", all_cols)

            if st.button("Run FPGrowth") and selected_cols:
                write_status(job_dir, "RUNNING", {"job": "FPGrowth", "cols": selected_cols})
                start_time = time.time()

                cols_expr = [concat(lit(c + "-"), col(c).cast("string")) for c in selected_cols]
                data = df.na.drop().withColumn("items", array(*cols_expr))

                fp = FPGrowth(itemsCol="items", minSupport=0.1, minConfidence=0.5)
                model = fp.fit(data)

                most_freq = model.freqItemsets.limit(15).toPandas()

                end_time = time.time()
                duration = end_time - start_time
                sp, eff = calculate_metrics("FPGrowth", num_nodes, duration)

                fp_path = save_pandas(most_freq, "fpgrowth_itemsets_top15.csv")
                meta_path = save_json(
                    {
                        "minSupport": 0.1,
                        "minConfidence": 0.5,
                        "execution_time_sec": duration,
                        "nodes_setting": num_nodes,
                        "speedup": sp,
                        "efficiency": eff,
                    },
                    "fpgrowth_meta.json"
                )

                write_status(job_dir, "FINISHED", {"itemsets_csv": fp_path, "meta_json": meta_path})

                st.subheader("Frequent Itemsets (Top 15)")
                st.table(most_freq)
                st.write(f"Execution Time: {duration:.4f} s | Speedup: {sp:.2f}x | Efficiency: {eff:.2f}")
                st.success("Saved to Drive ")
                st.caption(f"{fp_path}")
                st.caption(f"{meta_path}")


import os
import time
from pyngrok import ngrok


print(" Killing previous processes...")
os.system("pkill -f streamlit")
ngrok.kill()


if not os.path.exists("app.py"):
    print(" Error: file 'app.py' not found! Please run the previous cell to create the file.")
else:
    print(" app.py found.")

    print(" Starting Streamlit ")

    os.system("nohup streamlit run app.py --server.port 8501 --server.address 127.0.0.1 > streamlit.log 2>&1 &")

    for i in range(10):
        time.sleep(1)
        print(".", end="")
    print("\nChecking logs...")

    with open("streamlit.log", "r") as f:
        logs = f.read()


    if "Traceback" in logs or "Error" in logs or "Exception" in logs:
        print("\n Streamlit Failed to Start! Here is the error message:\n")
        print("==========================================")
        print(logs)
        print("==========================================")
        print(" Solution: Read the error above. It usually points to a missing library or syntax error in app.py.")

    elif "External URL" in logs or "Network URL" in logs or len(logs) > 0:
        print(" Streamlit started successfully in the background!")

        NGROK_AUTH_TOKEN = "37ppjr9zYMrVclRkJAqRUo9ysgp_84JypMzbU56ZLABxwtP6J"  # <--- ضع التوكن هنا

        try:
            ngrok.set_auth_token(NGROK_AUTH_TOKEN)
            public_url = ngrok.connect(8501).public_url
            print(f"\n SUCCESS! Click here: {public_url}")
        except Exception as e:
            print(f"\n Ngrok Error: {e}")

    else:
        print("\n Streamlit is taking too long or silent. Check 'streamlit.log' content manually.")
        print("Current log content:",logs)
