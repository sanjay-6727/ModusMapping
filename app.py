from flask import Flask, render_template, jsonify, Response, request, send_file
import polars as pl
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os
import gzip
import traceback
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from datetime import datetime
from werkzeug.utils import secure_filename
import tempfile
from collections import defaultdict
import re

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = tempfile.mkdtemp()
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Paths
csv_path = r"C:\Users\User\Desktop\Hackathon\crime_data.csv"
feather_path = r"C:\Users\User\Desktop\Hackathon\crime_data.feather"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def standardize_column_name(name):
    """Standardize column names for matching"""
    name = name.lower().strip()
    name = re.sub(r'[^a-z0-9]', '_', name)
    name = re.sub(r'_+', '_', name)
    return name

def find_column_mappings(dfs):
    """Find common columns across dataframes with fuzzy matching"""
    column_variations = defaultdict(list)
    
    # Collect all column names and their variations
    for df in dfs:
        for col in df.columns:
            std_col = standardize_column_name(col)
            column_variations[std_col].append(col)
    
    # Create mapping - use the most common variation for each standardized name
    column_mapping = {}
    for std_col, variations in column_variations.items():
        # Find the most common variation
        most_common = max(set(variations), key=variations.count)
        column_mapping[std_col] = most_common
        
        # Map all variations to the most common one
        for var in variations:
            if var != most_common:
                column_mapping[standardize_column_name(var)] = most_common
    
    return column_mapping

def merge_dataframes(dfs):
    """Merge multiple dataframes with column standardization"""
    if not dfs:
        return None
    
    # Find column mappings
    column_mapping = find_column_mappings(dfs)
    
    # Standardize columns in each dataframe
    standardized_dfs = []
    for df in dfs:
        # Rename columns to standardized names
        renamed_columns = {}
        for col in df.columns:
            std_col = standardize_column_name(col)
            if std_col in column_mapping:
                renamed_columns[col] = column_mapping[std_col]
        
        # Apply renaming and select only mapped columns
        df_std = df.rename(renamed_columns)
        standardized_dfs.append(df_std)
    
    # Concatenate all dataframes
    merged_df = pl.concat(standardized_dfs, how="diagonal")
    
    return merged_df

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({"error": "No files part"}), 400
    
    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({"error": "No selected files"}), 400
    
    # Read all uploaded CSV files
    dfs = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                df = pl.read_csv(filepath)
                dfs.append(df)
            except Exception as e:
                return jsonify({"error": f"Error reading {filename}: {str(e)}"}), 400
            finally:
                os.remove(filepath)  # Clean up
    
    if not dfs:
        return jsonify({"error": "No valid CSV files uploaded"}), 400
    
    # Merge dataframes
    try:
        merged_df = merge_dataframes(dfs)
        
        # Create response
        output = io.BytesIO()
        merged_df.write_csv(output)
        output.seek(0)
        
        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name='merged_crime_data.csv'
        )
    except Exception as e:
        return jsonify({"error": f"Error merging files: {str(e)}"}), 500

# Convert CSV to Feather (one-time setup)
if not os.path.exists(feather_path):
    print("Converting CSV to Feather...")
    try:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at {csv_path}")
        df = pl.read_csv(csv_path)
        df = df.with_columns(pl.col("crime_time").fill_null("00:00:00"))
        df = df.with_columns(
            pl.col("crime_time").str.strptime(pl.Time, "%H:%M:%S", strict=False)
            .dt.hour().fill_null(0).cast(pl.Int32).alias("crime_hour")
        )
        df.write_ipc(feather_path)
        print("Conversion complete.")
    except Exception as e:
        print(f"Error during conversion: {e}")
        traceback.print_exc()

# Load Feather file
lazy_df = None
data_loaded = False
try:
    lazy_df = pl.scan_ipc(feather_path).with_columns([
        pl.col("place_of_crime").fill_null("Unknown"),
        pl.col("crime_type").fill_null("Unknown"),
        pl.col("criminal_name").fill_null("Unknown"),
        pl.col("jail_entry_date").fill_null("N/A"),
        pl.col("bail_granted").fill_null("N/A"),
        pl.col("duration_of_imprisonment").fill_null("N/A"),
        pl.col("co_criminals").fill_null("None"),
        pl.col("motive").fill_null("Unknown")
    ])
    data_loaded = True
    print("Feather file loaded successfully.")
except Exception as e:
    print(f"Error loading Feather file: {e}")
    traceback.print_exc()

# City coordinates
city_coords = {
    "T. Nagar": [13.0418, 80.2335], "Velachery": [12.9823, 80.2209], "Anna Nagar": [13.0853, 80.2108],
    "Kodambakkam": [13.0526, 80.2215], "Adyar": [13.0067, 80.2553], "Tambaram": [12.9249, 80.1144],
    "Royapettah": [13.0556, 80.2648], "Vadapalani": [13.0516, 80.2124], "Perambur": [13.1181, 80.2451],
    "Mylapore": [13.0331, 80.2680], "Guindy": [13.0070, 80.2217], "Thiruvanmiyur": [12.9831, 80.2590],
    "Purasawalkam": [13.0885, 80.2615], "Washermanpet": [13.1186, 80.2822], "Nungambakkam": [13.0604, 80.2445]
}

def get_unique_values(column_name):
    if not data_loaded or lazy_df is None:
        return []
    try:
        return lazy_df.select(pl.col(column_name).unique().sort()).collect()[column_name].to_list()
    except Exception as e:
        print(f"Error getting unique values for {column_name}: {e}")
        return []

def prepare_data():
    if not data_loaded or lazy_df is None:
        raise Exception("Data not initialized.")

    # Calculate stats for the cards
    total_crimes = lazy_df.select(pl.len()).collect().item()
    arrest_rate = round(lazy_df.filter(pl.col("bail_granted") == "No").select(pl.len()).collect().item() / total_crimes * 100, 1)
    
    hotspot_threshold = lazy_df.group_by("place_of_crime").agg(pl.len().alias("count")).select(
        pl.col("count").quantile(0.75)
    ).collect().item()
    hotspot_areas = lazy_df.group_by("place_of_crime").agg(pl.len().alias("count")).filter(
        pl.col("count") > hotspot_threshold
    ).select(pl.len()).collect().item()
    
    recidivism_rate = round(lazy_df.group_by("criminal_name").agg(pl.len().alias("count")).filter(
        pl.col("count") > 1
    ).select(pl.len()).collect().item() / lazy_df.select(pl.col("criminal_name").n_unique()).collect().item() * 100, 1)

    stats = {
        "total_crimes": total_crimes,
        "arrest_rate": arrest_rate,
        "hotspot_areas": hotspot_areas,
        "recidivism_rate": recidivism_rate
    }

    # Map Heatmap Data with last incident date
    map_data_lazy = lazy_df.group_by("place_of_crime").agg([
        pl.len().alias("count"),
        pl.col("crime_time").max().alias("last_incident")
    ]).with_columns([
        pl.col("place_of_crime").map_elements(lambda x: city_coords.get(x, [13.0827, 80.2707])[0], return_dtype=pl.Float64).alias("lat"),
        pl.col("place_of_crime").map_elements(lambda x: city_coords.get(x, [13.0827, 80.2707])[1], return_dtype=pl.Float64).alias("lon")
    ]).select(["place_of_crime", "count", "lat", "lon", "last_incident"])
    map_data = map_data_lazy.collect().to_dicts()

    # Bar Chart Data
    bar_counts_lazy = lazy_df.group_by("crime_hour").agg(pl.len().alias("count")).sort("crime_hour")
    bar_counts_df = bar_counts_lazy.collect()
    bar_counts = np.zeros(24, dtype=int)
    for time, count in zip(bar_counts_df["crime_hour"], bar_counts_df["count"]):
        bar_counts[time] = count
    bar_data = {"hours": list(range(24)), "counts": bar_counts.tolist()}

    # Pie Chart Data (all locations)
    pie_data_lazy = lazy_df.group_by("place_of_crime").agg(pl.len().alias("count")).sort("count", descending=True)
    pie_df = pie_data_lazy.collect()
    pie_data = {"labels": pie_df["place_of_crime"].to_list(), "counts": pie_df["count"].to_list()}

    # Line Chart Data (all crime types)
    crime_type_trends_lazy = lazy_df.group_by(["crime_hour", "crime_type"]).agg(pl.len().alias("count"))
    pivot_crime_trends = crime_type_trends_lazy.collect().pivot(
        values="count", index="crime_hour", on="crime_type", aggregate_function="sum"
    ).fill_null(0)
    line_df = pivot_crime_trends.select(pl.all().exclude("crime_hour"))
    line_data = {
        "hours": pivot_crime_trends["crime_hour"].to_list(),
        "datasets": [{"label": col, "data": line_df[col].to_list()} for col in line_df.columns]
    }

    # Criminal History Data (limited for performance)
    criminal_history = lazy_df.select([
        "criminal_name", "crime_type", "crime_time", "crime_hour", "place_of_crime",
        "jail_entry_date", "bail_granted", "duration_of_imprisonment", "co_criminals", "motive"
    ]).limit(1000).collect().to_dicts()

    # Recidivism Data (top 10)
    recidivism_data_lazy = lazy_df.group_by("criminal_name").agg(pl.len().alias("crime_count")).filter(pl.col("crime_count") > 1).sort("crime_count", descending=True).limit(10)
    recidivism_data = recidivism_data_lazy.collect().to_dicts()

    return {
        "stats": stats,
        "map_heatmap": map_data,
        "bar": bar_data,
        "pie": pie_data,
        "line": line_data,
        "criminal_history": criminal_history,
        "recidivism": recidivism_data
    }

@app.route("/api/criminal-history")
def get_criminal_history():
    if not data_loaded or lazy_df is None:
        return jsonify({"error": "Data source not initialized"}), 500
    try:
        draw = int(request.args.get("draw", 1))
        start = int(request.args.get("start", 0))
        length = int(request.args.get("length", 10))
        search_value = request.args.get("search[value]", "").strip()
        crime_type = request.args.get("crime_type", "").strip()
        motive = request.args.get("motive", "").strip()
        place = request.args.get("place", "").strip()

        query = lazy_df.select([
            "criminal_name", "crime_type", "date_of_crime", "crime_time", "place_of_crime",
            "bail_granted", "motive", "co_criminals"
        ])

        # Apply filters
        if crime_type:
            query = query.filter(pl.col("crime_type") == crime_type)
        if motive:
            query = query.filter(pl.col("motive") == motive)
        if place:
            query = query.filter(pl.col("place_of_crime") == place)
        if search_value:
            try:
                search_int = int(search_value)
                query = query.filter(
                    pl.any_horizontal(pl.col("*").cast(pl.Utf8).str.contains(search_value, literal=True)) |
                    (pl.col("crime_hour") == search_int)
                )
            except ValueError:
                query = query.filter(pl.any_horizontal(pl.col("*").cast(pl.Utf8).str.contains(search_value, literal=True)))

        total_records = lazy_df.select(pl.len()).collect().item()
        filtered_count = query.select(pl.len()).collect().item()
        paginated_data = query.collect().slice(start, length).to_dicts()

        # Format the data for the table
        formatted_data = []
        for row in paginated_data:
            formatted_row = {
                "criminal_name": row["criminal_name"],
                "crime_type": row["crime_type"],
                "date_of_crime": row["date_of_crime"],  # New field
                "crime_time": row["crime_time"],
                "place_of_crime": row["place_of_crime"],
                "co_criminals": row["co_criminals"],
                "bail_granted": "On Bail" if row["bail_granted"] == "Yes" else "In Custody",
                "actions": ""  # This will be handled by the frontend
            }
            formatted_data.append(formatted_row)

        return jsonify({
            "draw": draw,
            "recordsTotal": total_records,
            "recordsFiltered": filtered_count,
            "data": formatted_data
        })
    except Exception as e:
        print(f"Error in /api/criminal-history: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/filters")
def get_filters():
    try:
        return jsonify({
            "crime_types": get_unique_values("crime_type"),
            "motives": get_unique_values("motive"),
            "places": get_unique_values("place_of_crime")
        })
    except Exception as e:
        print(f"Error in /api/filters: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/export", methods=["GET"])
def export_criminals():
    if not data_loaded or lazy_df is None:
        return jsonify({"error": "Data source not initialized"}), 500
    try:
        def generate():
            headers = ["criminal_name", "crime_type", "crime_time", "crime_hour", "place_of_crime",
                       "jail_entry_date", "bail_granted", "duration_of_imprisonment", "co_criminals", "motive"]
            yield ",".join(headers) + "\n"
            for row in lazy_df.select(headers).collect().iter_rows():
                yield ",".join(str(value) if value is not None else "N/A" for value in row) + "\n"

        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode="wb") as gz:
            gz.write("".join(generate()).encode())
        buffer.seek(0)
        return Response(buffer, mimetype="application/gzip", headers={
            "Content-Disposition": "attachment; filename=criminal_history.csv.gz"
        })
    except Exception as e:
        print(f"Error in /export: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/export_pdf", methods=["GET"])
def export_criminals_pdf():
    if not data_loaded or lazy_df is None:
        return jsonify({"error": "Data source not initialized"}), 500
    try:
        headers = ["Name", "Crime Type", "Time", "Location", "Co-Criminals", "Status"]
        df = lazy_df.select([
            "criminal_name", "crime_type", "crime_time", "place_of_crime", "co_criminals", "bail_granted"
        ]).collect()

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []

        # Add title
        from reportlab.platypus import Paragraph
        from reportlab.lib.styles import getSampleStyleSheet
        styles = getSampleStyleSheet()
        title = Paragraph("ModOps - Criminal History Report", styles['Title'])
        elements.append(title)
        
        # Add date
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        date = Paragraph(f"Generated on: {date_str}", styles['Normal'])
        elements.append(date)
        elements.append(Paragraph("<br/><br/>", styles['Normal']))

        chunk_size = 50  # Smaller chunks for PDF
        for i in range(0, df.height, chunk_size):
            chunk = df.slice(i, chunk_size)
            data = [headers] + [[
                str(row["criminal_name"]),
                str(row["crime_type"]),
                str(row["crime_time"]),
                str(row["place_of_crime"]),
                str(row["co_criminals"]),
                "On Bail" if row["bail_granted"] == "Yes" else "In Custody"
            ] for row in chunk.iter_rows(named=True)]
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2a52be")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#f8f9fa")),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")])
            ]))
            elements.append(table)
            elements.append(Paragraph("<br/>", styles['Normal']))

        doc.build(elements)
        buffer.seek(0)
        return Response(buffer, mimetype="application/pdf", headers={
            "Content-Disposition": "attachment; filename=ModOps_criminal_history.pdf"
        })
    except Exception as e:
        print(f"Error in /export_pdf: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/crime-data")
def get_crime_data():
    try:
        data = prepare_data()
        return jsonify(data)
    except Exception as e:
        print(f"Error in /api/crime-data: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
