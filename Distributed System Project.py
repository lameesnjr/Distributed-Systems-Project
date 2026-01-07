%%writefile app.py
import streamlit as st
import pandas as pd
import time
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import array, col, lit, concat
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor
from pyspark.ml.fpm import FPGrowth

# محاولة استيراد مكتبة PDF
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

# تهيئة الصفحة والذاكرة
st.set_page_config(page_title="Cloud Data Service", layout="wide")

# نحتاج لتخزين أوقات الأساس (Baseline) لحساب التسريع (Speedup)
if 'baseline_times' not in st.session_state:
    st.session_state['baseline_times'] = {}

st.title(" Cloud-Based Distributed Data Processing Service")

# دالة مساعدة لحساب الأداء
def calculate_metrics(job_name, nodes, duration):
    if nodes == 1:
        st.session_state['baseline_times'][job_name] = duration
        speedup = 1.0
        efficiency = 1.0
    else:
        t1 = st.session_state['baseline_times'].get(job_name)
        if t1:
            speedup = t1 / duration
            efficiency = speedup / nodes
        else:
            speedup = 0.0
            efficiency = 0.0
            
    return speedup, efficiency

# رفع البيانات
uploaded_file = st.file_uploader("1. Upload Dataset", type=["csv", "json", "txt", "pdf"])

if uploaded_file is not None:
    file_name = uploaded_file.name
    file_extension = file_name.split('.')[-1].lower()
    temp_filename = f"temp_data.{file_extension}"
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f" {file_extension.upper()} File uploaded!")

    # إعدادات النظام الموزع
    st.sidebar.header("Processing Options")
    num_nodes = st.sidebar.select_slider("Select Number of Worker Nodes", options=[1, 2, 4, 8])
    
    # تهيئة Spark
    spark = SparkSession.builder \
        .appName("EngineeringProject") \
        .master(f"local[{num_nodes}]") \
        .getOrCreate()
    
    # قراءة الملف
    df = None
    try:
        if file_extension == 'csv':
            df = spark.read.csv(temp_filename, header=True, inferSchema=True)
        elif file_extension == 'json':
            df = spark.read.option("multiline", "true").json(temp_filename)
        elif file_extension == 'txt':
            df = spark.read.csv(temp_filename, header=True, inferSchema=True, sep="\t")
        elif file_extension == 'pdf':
            if PdfReader:
                reader = PdfReader(temp_filename)
                text_content = []
                for page in reader.pages:
                    text_content.append((page.extract_text(),))
                df = spark.createDataFrame(text_content, ["pdf_content"])
                st.warning(" PDF is text-only.")
            else:
                st.error("Please install pypdf: `!pip install pypdf`")
    except Exception as e:
        st.error(f"Error reading file: {e}")

    if df is not None:
        numeric_cols = [c for c, t in df.dtypes if t in ('int', 'double', 'float')]
        all_cols = df.columns

        # قائمة الوظائف 
        job_type = st.selectbox(
            "2. Select Processing Job",
            [
                "Descriptive Statistics", 
                "K-Means Clustering", 
                "Linear Regression", 
                "Decision Tree (Regression)",
                "FPGrowth (Frequent Patterns)"
            ]
        )

        # 1. Statistics
        if job_type == "Descriptive Statistics":
            if st.button("Run Analysis"):
                start_time = time.time()
                if not numeric_cols:
                    st.warning("No numeric columns.")
                    stats = pd.DataFrame({"Count": [df.count()]})
                else:
                    stats = df.describe().toPandas()
                end_time = time.time()
                duration = end_time - start_time
                sp, eff = calculate_metrics("Statistics", num_nodes, duration)

                st.subheader(" Results Table")
                st.table(stats)
                st.write(f"**Execution Time:** {duration:.4f} s | **Speedup:** {sp:.2f}x | **Efficiency:** {eff:.2f}")

        # 2. K-Means
        elif job_type == "K-Means Clustering":
            if not numeric_cols: st.error("Requires numeric columns.")
            else:
                selected_cols = st.multiselect("Select Columns", numeric_cols, default=numeric_cols[:2] if len(numeric_cols)>1 else None)
                k_val = st.slider("K Clusters", 2, 10, 3)
                if st.button("Run Clustering") and len(selected_cols) >= 2:
                    start_time = time.time()
                    assembler = VectorAssembler(inputCols=selected_cols, outputCol="features")
                    data = assembler.transform(df.na.drop())
                    kmeans = KMeans().setK(k_val).setSeed(1)
                    model = kmeans.fit(data)
                    end_time = time.time()
                    duration = end_time - start_time
                    sp, eff = calculate_metrics("K-Means", num_nodes, duration)

                    st.success("Done.")
                    st.subheader(" Cluster Centers (Table)")
                    centers = model.clusterCenters()
                    center_df = pd.DataFrame(centers, columns=selected_cols)
                    center_df.index.name = "Cluster ID"
                    st.table(center_df)
                    st.write(f"**Execution Time:** {duration:.4f} s | **Speedup:** {sp:.2f}x | **Efficiency:** {eff:.2f}")

        # 3. Linear Regression (Coefficients Output)
        elif job_type == "Linear Regression":
            if not numeric_cols: st.error("Requires numeric columns.")
            else:
                target_col = st.selectbox("Target (Y)", numeric_cols)
                feature_cols = st.multiselect("Features (X)", [c for c in numeric_cols if c != target_col])
                if st.button("Train Model") and feature_cols:
                    start_time = time.time()
                    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
                    data = assembler.transform(df.na.drop())
                    
                    lr = LinearRegression(labelCol=target_col)
                    model = lr.fit(data)
                    
                    end_time = time.time()
                    duration = end_time - start_time
                    sp, eff = calculate_metrics("Linear Regression", num_nodes, duration)

                    st.success("Model Trained Successfully.")
                    
                    st.subheader(" Model Coefficients")
                    st.write("The weights assigned to each feature:")
                    
                    coefficients = model.coefficients.toArray()
                    coeff_df = pd.DataFrame({
                        "Feature": feature_cols,
                        "Coefficient": coefficients
                    })
                    
                    st.table(coeff_df)
                    st.write(f"**Model Intercept:** {model.intercept:.4f}")
                    
                    st.write(f"**Execution Time:** {duration:.4f} s | **Speedup:** {sp:.2f}x | **Efficiency:** {eff:.2f}")

        # 4. Decision Tree (Prediction Table Output)
        elif job_type == "Decision Tree (Regression)":
            if not numeric_cols: st.error("Requires numeric columns.")
            else:
                target_col = st.selectbox("Target (Y)", numeric_cols, key="dt_target")
                feature_cols = st.multiselect("Features (X)", [c for c in numeric_cols if c != target_col], key="dt_features")
                max_depth = st.slider("Max Tree Depth", 2, 5, 3)

                if st.button("Train Decision Tree") and feature_cols:
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

                    st.success("Model Trained Successfully.")
                    
                    st.subheader(" Prediction Results")
                    
                    display_df = predictions.select(feature_cols + ['prediction']).limit(15).toPandas()
                    display_df.rename(columns={'prediction': f'Predicted {target_col}'}, inplace=True)
                    st.table(display_df)
                    
                    st.write(f"**Execution Time:** {duration:.4f} s | **Speedup:** {sp:.2f}x | **Efficiency:** {eff:.2f}")

        # 5. FPGrowth (Frequent Patterns Only)
        elif job_type == "FPGrowth (Frequent Patterns)":
            selected_cols = st.multiselect("Select Categorical Columns", all_cols)
            
            if st.button("Run FPGrowth") and selected_cols:
                start_time = time.time()
                cols_expr = [concat(lit(c + "-"), col(c).cast("string")) for c in selected_cols]
                data = df.na.drop().withColumn("items", array(*cols_expr))
                
                fp = FPGrowth(itemsCol="items", minSupport=0.1, minConfidence=0.5)
                model = fp.fit(data)
                
                most_freq = model.freqItemsets.limit(15).toPandas()
                
                end_time = time.time()
                duration = end_time - start_time
                sp, eff = calculate_metrics("FPGrowth", num_nodes, duration)

                st.subheader(" Frequent Itemsets (Top 15)")
                st.table(most_freq)
                
                st.write(f"**Execution Time:** {duration:.4f} s | **Speedup:** {sp:.2f}x | **Efficiency:** {eff:.2f}")
