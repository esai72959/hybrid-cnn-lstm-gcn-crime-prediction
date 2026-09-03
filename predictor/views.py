import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import csv
from django.conf import settings
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from predictor.services.dataset_loader import dataset_loader

logger = logging.getLogger(__name__)


# ============================================================
# MODEL PERFORMANCE DATA (single authoritative source)
# ============================================================
# These figures are the held-out test-set results produced by the
# evaluate_model() step of cnn_model.py, lstm_model.py and
# hybrid_model.py. This dict is the ONLY place they are defined -
# performance.html and performance.js both read them from the view's
# context (see `performance()` below), so the KPI cards, the
# evaluation table and the accuracy chart can never disagree again.
#
# "mse" is not saved separately by the training scripts, so it is
# derived here as rmse ** 2 (MSE is RMSE squared by definition) rather
# than re-measured or invented.
_DISTRICT_NEIGHBORS_CACHE: Optional[Dict[str, Any]] = None

MODEL_PERFORMANCE_METRICS = {
    "cnn": {
        "key": "cnn",
        "label": "CNN",
        "full_name": "Convolutional Neural Network",
        "r2": 0.8418,
        "accuracy": 84.18,
        "rmse": 0.2145,
        "mae": 0.1687,
        "status": "Baseline (Single Split)",
    },
    "lstm": {
        "key": "lstm",
        "label": "LSTM",
        "full_name": "Long Short-Term Memory Network",
        "r2": 0.8818,
        "accuracy": 88.18,
        "rmse": 0.1892,
        "mae": 0.1421,
        "status": "Baseline (Single Split)",
    },
    "hybrid": {
        "key": "hybrid",
        "label": "Hybrid CNN-LSTM (2-Way)",
        "full_name": "Hybrid CNN-LSTM Architecture",
        "r2": 0.9647,
        "accuracy": 96.47,
        "rmse": 583.19,
        "mae": 311.24,
        "status": "5-Fold CV (96.47% ± 1.71%, not statistically significant vs. 3-way)",
    },
    "hybrid_gcn": {
        "key": "hybrid_gcn",
        "label": "Hybrid CNN-LSTM-GCN (3-Way)",
        "full_name": "Hybrid Spatio-Temporal Graph Architecture",
        "r2": 0.9610,
        "accuracy": 96.10,
        "rmse": 612.47,
        "mae": 337.54,
        "status": "5-Fold CV (96.10% ± 2.00%, not statistically significant vs. 2-way)",
    },
}

for _model in MODEL_PERFORMANCE_METRICS.values():
    _model["mse"] = round(_model["rmse"] ** 2, 4)


def _get_hybrid_training_history():
    """
    Reads results/hybrid_training_history.csv, written by
    hybrid_model.py's save_training_history() after an actual training
    run. Returns None if the file doesn't exist yet or is malformed, so
    the Performance page can skip the loss chart instead of showing
    invented epoch values.
    """
    history_path = Path(settings.BASE_DIR) / "results" / "hybrid_training_history.csv"

    if not history_path.exists():
        return None

    try:
        history_df = pd.read_csv(history_path)
    except Exception:
        logger.warning("Could not read hybrid training history at %s", history_path)
        return None

    if "loss" not in history_df.columns or "val_loss" not in history_df.columns or history_df.empty:
        return None

    return {
        "epochs": [f"Epoch {i + 1}" for i in range(len(history_df))],
        "training_loss": [round(float(v), 4) for v in history_df["loss"]],
        "validation_loss": [round(float(v), 4) for v in history_df["val_loss"]],
    }


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_state_name(state_name):
    """Normalize state names for clean display while keeping matching flexible."""
    if not state_name:
        return ""
    
    clean = str(state_name).strip()
    
    replacements = {
        "A&N ISLANDS": "A & N ISLANDS",
        "D&N HAVELI": "D & N HAVELI",
        "D & N HAVELI": "D & N HAVELI",
        "DAMAN & DIU": "DAMAN & DIU",
        "DAMAN AND DIU": "DAMAN & DIU",
    }
    
    return replacements.get(clean.upper(), clean)


def _get_yearly_crime_trend():
    """Helper to fetch yearly crime trend data across the entire dataset."""
    df = dataset_loader.get_dataframe()
    if df is None or df.empty or "YEAR" not in df.columns or "TOTAL IPC CRIMES" not in df.columns:
        return []

    yearly = df.groupby("YEAR")["TOTAL IPC CRIMES"].sum()
    return [{"year": int(year), "total_crimes": int(val)} for year, val in yearly.items()]


def _get_recommendation_for_risk(risk_level: str):
    """Return (recommendation_text, risk_score) for a given risk level.

    Handles the four risk levels produced by CrimePredictor: "Low", "Moderate", "High", "Very High".
    """
    if risk_level == "Very High":
        return (
            "Deploy immediate additional patrols and coordinate with local "
            "law enforcement on targeted intervention in this district.",
            95,
        )
    elif risk_level == "High":
        return (
            "Increase police patrols and surveillance in identified hotspots.",
            90,
        )
    elif risk_level == "Moderate":
        return (
            "Continuous monitoring and preventive measures are recommended.",
            60,
        )
    else:  # "Low"
        return (
            "Current crime trends appear stable. Continue regular monitoring.",
            30,
        )


# ============================================================
# CORE APPLICATION PAGES
# ============================================================

def home(request):
    return render(request, "home.html")



def dashboard(request):
    return render(request, "dashboard.html")



def prediction(request):
    raw_states = dataset_loader.get_states() or []
    normalized_states = sorted(list(set(normalize_state_name(s) for s in raw_states if s)))
    years = list(range(2001, 2031))

    crime_types = [
        "TOTAL IPC CRIMES",
        "MURDER",
        "RAPE",
        "ROBBERY",
        "BURGLARY",
        "THEFT",
        "RIOTS",
        "ATTEMPT TO MURDER",
    ]

    context = {
        "states": normalized_states,
        "years": years,
        "crime_types": crime_types,
    }

    return render(request, "prediction.html", context)



def forecast(request):
    """GET /forecast/

    Renders the dedicated Forecast page shell template.
    All state and district options and forecast payload loading occur client-side.
    """
    return render(request, "forecast.html")



def performance(request):
    df = dataset_loader.get_dataframe()

    dataset_records = 0
    dataset_period = "N/A"

    if df is not None and not df.empty:
        dataset_records = len(df)

        if "YEAR" in df.columns:
            numeric_years = pd.to_numeric(df["YEAR"], errors="coerce").dropna()
            if not numeric_years.empty:
                dataset_period = f"{int(numeric_years.min())}\u2013{int(numeric_years.max())}"

    loss_history = _get_hybrid_training_history()

    # Load 5-Fold Cross-Validation Benchmark Data
    cv_data_path = Path(settings.BASE_DIR) / "results" / "cv_5fold_results.json"
    cv_benchmark_data = {}
    if cv_data_path.exists():
        try:
            with open(cv_data_path, "r", encoding="utf-8") as f:
                cv_benchmark_data = json.load(f)
        except Exception:
            cv_benchmark_data = {}

    chart_payload = {
        "labels": [
            MODEL_PERFORMANCE_METRICS["cnn"]["label"],
            MODEL_PERFORMANCE_METRICS["lstm"]["label"],
            MODEL_PERFORMANCE_METRICS["hybrid"]["label"],
        ],
        "accuracy": [
            MODEL_PERFORMANCE_METRICS["cnn"]["accuracy"],
            MODEL_PERFORMANCE_METRICS["lstm"]["accuracy"],
            MODEL_PERFORMANCE_METRICS["hybrid"]["accuracy"],
        ],
        "loss_history": loss_history,
    }

    # Load Scatter Evaluation Points Data
    scatter_data_path = Path(settings.BASE_DIR) / "predictor" / "static" / "data" / "eval_scatter_data.json"
    scatter_points = []
    if scatter_data_path.exists():
        try:
            with open(scatter_data_path, "r", encoding="utf-8") as f:
                s_data = json.load(f)
                scatter_points = s_data.get("scatter_points", [])
        except Exception:
            scatter_points = []

    context = {
        "metrics": MODEL_PERFORMANCE_METRICS,
        "dataset_records": dataset_records,
        "dataset_records_display": f"{dataset_records:,}" if dataset_records else "N/A",
        "dataset_period": dataset_period,
        "chart_payload": chart_payload,
        "cv_benchmark_data": cv_benchmark_data,
        "scatter_points": scatter_points,
    }

    return render(request, "performance.html", context)


# ============================================================
# DATASET PAGE
# ============================================================


def dataset(request):
    query = request.GET.get("q", "").strip()
    selected_state = request.GET.get("state", "").strip()
    selected_district = request.GET.get("district", "").strip()
    selected_year = request.GET.get("year", "").strip()
    page_number = request.GET.get("page", 1)

    all_states = []
    districts = []
    years = []

    total_records = 0
    total_states = 0
    total_districts = 0
    year_range_str = "N/A"
    total_features = 33
    dataset_status = "Active"

    stats = {
        "avg_crime_count": "Not available",
        "highest_volume_region": "Not available",
        "lowest_volume_region": "Not available",
        "growth_trend": "Not available",
    }

    df = dataset_loader.get_dataframe()

    if df is not None and not df.empty:
        total_records = len(df)

        try:
            raw_states = dataset_loader.get_states() or []
            all_states = sorted(list(set(normalize_state_name(s) for s in raw_states if s)))
            total_states = len(all_states)
        except Exception:
            all_states = []
            total_states = 0

        if selected_state:
            try:
                districts = dataset_loader.get_districts(selected_state)
            except Exception:
                districts = []

        if "DISTRICT" in df.columns:
            total_districts = df["DISTRICT"].dropna().astype(str).str.strip().nunique()

        if "YEAR" in df.columns:
            numeric_years = pd.to_numeric(df["YEAR"], errors="coerce").dropna()
            years = sorted(numeric_years.astype(int).unique().tolist(), reverse=True)
            if years:
                year_range_str = f"{min(years)}–{max(years)}"

        if "TOTAL IPC CRIMES" in df.columns:
            crime_series = pd.to_numeric(df["TOTAL IPC CRIMES"], errors="coerce").dropna()

            if not crime_series.empty:
                average_crime_count = crime_series.mean()
                stats["avg_crime_count"] = f"{round(average_crime_count):,}"

            if "STATE/UT" in df.columns and "DISTRICT" in df.columns:
                region_df = df[["STATE/UT", "DISTRICT", "TOTAL IPC CRIMES"]].copy()
                region_df["TOTAL IPC CRIMES"] = pd.to_numeric(region_df["TOTAL IPC CRIMES"], errors="coerce")
                region_df = region_df.dropna(subset=["STATE/UT", "DISTRICT", "TOTAL IPC CRIMES"])

                if not region_df.empty:
                    grouped = region_df.groupby(["STATE/UT", "DISTRICT"])["TOTAL IPC CRIMES"].sum()
                    if not grouped.empty:
                        highest_region = grouped.idxmax()
                        lowest_region = grouped.idxmin()

                        stats["highest_volume_region"] = f"{highest_region[1]} ({highest_region[0]})"
                        stats["lowest_volume_region"] = f"{lowest_region[1]} ({lowest_region[0]})"

            if "YEAR" in df.columns:
                trend_df = df[["YEAR", "TOTAL IPC CRIMES"]].copy()
                trend_df["YEAR"] = pd.to_numeric(trend_df["YEAR"], errors="coerce")
                trend_df["TOTAL IPC CRIMES"] = pd.to_numeric(trend_df["TOTAL IPC CRIMES"], errors="coerce")
                trend_df = trend_df.dropna(subset=["YEAR", "TOTAL IPC CRIMES"])

                if not trend_df.empty:
                    earliest_year = int(trend_df["YEAR"].min())
                    latest_year = int(trend_df["YEAR"].max())

                    earliest_values = trend_df[trend_df["YEAR"] == earliest_year]["TOTAL IPC CRIMES"]
                    latest_values = trend_df[trend_df["YEAR"] == latest_year]["TOTAL IPC CRIMES"]

                    if not earliest_values.empty and not latest_values.empty:
                        earliest_average = earliest_values.mean()
                        latest_average = latest_values.mean()

                        if earliest_average > 0:
                            percentage_change = ((latest_average - earliest_average) / earliest_average) * 100
                            sign = "+" if percentage_change >= 0 else ""
                            stats["growth_trend"] = f"{sign}{percentage_change:.1f}% ({earliest_year}–{latest_year})"

        filtered_df = df.copy()

        if query:
            query_lower = query.lower()
            state_mask = False
            district_mask = False

            if "STATE/UT" in filtered_df.columns:
                state_mask = filtered_df["STATE/UT"].astype(str).str.lower().str.contains(query_lower, na=False)

            if "DISTRICT" in filtered_df.columns:
                district_mask = filtered_df["DISTRICT"].astype(str).str.lower().str.contains(query_lower, na=False)

            filtered_df = filtered_df[state_mask | district_mask]

        if selected_state and "STATE/UT" in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df["STATE/UT"].astype(str).str.strip().str.upper() == selected_state.upper()
            ]

        if selected_district and "DISTRICT" in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df["DISTRICT"].astype(str).str.strip().str.upper() == selected_district.upper()
            ]

        if selected_year and "YEAR" in filtered_df.columns:
            try:
                year_value = int(selected_year)
                filtered_df = filtered_df[pd.to_numeric(filtered_df["YEAR"], errors="coerce") == year_value]
            except ValueError:
                pass

        cols_to_display = ["STATE/UT", "DISTRICT", "YEAR"]
        candidate_columns = [
            "YEAR_INDEX",
            "TOTAL IPC CRIMES",
            "MURDER",
            "RAPE",
            "ROBBERY",
            "BURGLARY",
            "THEFT",
            "RIOTS",
        ]

        for column in candidate_columns:
            if column in filtered_df.columns and column not in cols_to_display:
                cols_to_display.append(column)
            if len(cols_to_display) >= 6:
                break

        if set(cols_to_display).issubset(filtered_df.columns):
            records_list = filtered_df[cols_to_display].to_dict(orient="records")
        else:
            records_list = filtered_df.to_dict(orient="records")

    else:
        records_list = []
        cols_to_display = ["STATE/UT", "DISTRICT", "YEAR"]

    paginator = Paginator(records_list, 15)

    try:
        page_obj = paginator.page(page_number)
    except PageNotAnInteger:
        page_obj = paginator.page(1)
    except EmptyPage:
        page_obj = paginator.page(paginator.num_pages if paginator.num_pages > 0 else 1)

    table_headers = [column.replace("_", " ").title() for column in cols_to_display]
    page_records = []

    for row in page_obj:
        formatted_row = []
        for column in cols_to_display:
            value = row.get(column, "-")
            if column == "TOTAL IPC CRIMES":
                try:
                    value = f"{float(value):,.0f}"
                except (ValueError, TypeError):
                    pass
            formatted_row.append(value)
        page_records.append(formatted_row)

    context = {
        "overview": {
            "total_records": total_records,
            "total_states": total_states,
            "total_districts": total_districts,
            "year_range": year_range_str,
            "total_features": total_features,
            "status": dataset_status,
        },
        "stats": stats,
        "all_states": all_states,
        "districts": districts,
        "years": years,
        "selected_q": query,
        "selected_state": selected_state,
        "selected_district": selected_district,
        "selected_year": selected_year,
        "table_headers": table_headers,
        "page_records": page_records,
        "page_obj": page_obj,
        "dataset_records": page_records,
    }

    return render(request, "dataset.html", context)


# ============================================================
# OTHER APPLICATION PAGES
# ============================================================


def about(request):
    return render(request, "about.html")
def problem_objectives(request):
    return render(request, 'problem_objectives.html')
def methodology(request):
    return render(request, 'methodology.html')



def contact(request):
    return render(request, "contact.html")


def login_view(request):
    return render(request, "login.html")


def signup_view(request):
    return render(request, "signup.html")


def forgot_password(request):
    return render(request, "forgot_password.html")



def _get_crime_predictor():
    """Return the existing project CrimePredictor service."""
    try:
        from predictor.services.predictor import crime_predictor
        return crime_predictor
    except Exception:
        try:
            from predictor.services.predictor import CrimePredictor
            return CrimePredictor()
        except Exception:
            return None


def _prediction_result_to_dict(result):
    """Normalize predictor results for Django JSON responses."""
    if isinstance(result, dict):
        return result
    if result is None:
        return {}

    result_dict = {}
    for key in (
        "predicted_crime_count",
        "predicted_count",
        "risk_level",
        "risk_score",
        "confidence",
        "recommendation",
        "model",
        "state",
        "district",
        "year",
        "dataset_year",
    ):
        if hasattr(result, key):
            result_dict[key] = getattr(result, key)
    return result_dict


def _get_yearly_crime_trend_for_location(state, district):
    """Return real historical crime values for one selected location."""
    df = dataset_loader.get_dataframe()
    if (
        df is None
        or df.empty
        or "YEAR" not in df.columns
        or "TOTAL IPC CRIMES" not in df.columns
    ):
        return []

    state_col = "STATE/UT" if "STATE/UT" in df.columns else "State/UT"
    district_col = "DISTRICT" if "DISTRICT" in df.columns else "District"

    filtered = df[
        (df[state_col].astype(str).str.strip().str.upper() == state.upper())
        & (
            df[district_col].astype(str).str.strip().str.upper()
            == district.upper()
        )
    ].copy()

    if filtered.empty:
        return []

    filtered["YEAR"] = pd.to_numeric(filtered["YEAR"], errors="coerce")
    filtered["TOTAL IPC CRIMES"] = pd.to_numeric(
        filtered["TOTAL IPC CRIMES"], errors="coerce"
    )
    filtered = filtered.dropna(
        subset=["YEAR", "TOTAL IPC CRIMES"]
    ).sort_values("YEAR")

    return [
        {"year": int(year), "total_crimes": int(value)}
        for year, value in zip(
            filtered["YEAR"], filtered["TOTAL IPC CRIMES"]
        )
    ]


# ============================================================
# API ENDPOINTS
# ============================================================


def api_states(request):
    try:
        states = dataset_loader.get_states() or []
        normalized_states = sorted(list(set(normalize_state_name(s) for s in states if s and str(s).strip())), key=str.upper)

        return JsonResponse({
            "status": "success",
            "states": normalized_states
        })
    except Exception as error:
        return JsonResponse({"status": "error", "states": [], "message": str(error)}, status=500)



def api_districts(request):
    state = request.GET.get("state", "").strip()
    if not state:
        return JsonResponse({"districts": []})

    districts = dataset_loader.get_districts(state)
    return JsonResponse({"districts": districts})


def _check_hybrid_model_loaded():
    """
    Best-effort check for whether the trained Hybrid CNN-LSTM model can
    currently be loaded into memory.

    `dataset_loader` deliberately never touches TensorFlow/Keras (see
    dataset_loader.py), so any real model-loading service lives
    elsewhere in the project. We probe for it defensively: if no such
    service is wired up yet, or importing/loading it fails for any
    reason (e.g. a TensorFlow/Keras compatibility issue), we report the
    model as unavailable rather than assuming it is ready.
    """
    try:
        from predictor.services.model_loader import model_loader  # optional service
    except Exception:
        return False

    try:
        return bool(model_loader.is_loaded())
    except Exception:
        return False



def api_dashboard_data(request):
    df = dataset_loader.get_dataframe()
    if df is None or df.empty:
        return JsonResponse({"error": "Dataset not loaded"}, status=500)

    total_records = len(df)
    total_states = df["STATE/UT"].nunique() if "STATE/UT" in df.columns else 0
    total_districts = df["DISTRICT"].nunique() if "DISTRICT" in df.columns else 0

    chart_data = {}
    if "YEAR" in df.columns and "TOTAL IPC CRIMES" in df.columns:
        yearly = df.groupby("YEAR")["TOTAL IPC CRIMES"].sum()
        chart_data = {str(year): int(value) for year, value in yearly.items()}

    return JsonResponse({
        "total_records": total_records,
        "total_states": total_states,
        "total_districts": total_districts,
        "yearly_crime_trend": chart_data,
    })


def api_dashboard(request):
    """GET /api/dashboard/

    The single, authoritative Dashboard API. Reads dataset stats from
    dataset_loader, model performance from MODEL_PERFORMANCE_METRICS
    (the same source performance() uses, so the two pages can never
    disagree), and real system status - and returns them in the one
    response structure dashboard.js expects.

    Never fabricates values: any field that can't be honestly computed
    is returned as null so the frontend renders "Data unavailable"
    instead of a fake 0.
    """
    try:
        df = dataset_loader.get_dataframe()
        dataset_loaded = df is not None and not df.empty

        records = None
        period = None

        if dataset_loaded:
            records = int(len(df))

            if "YEAR" in df.columns:
                numeric_years = pd.to_numeric(df["YEAR"], errors="coerce").dropna()
                if not numeric_years.empty:
                    period = f"{int(numeric_years.min())}\u2013{int(numeric_years.max())}"

        # 33 model-ready spatial + temporal features used by the research
        # methodology (same constant the Dataset page already uses). This
        # isn't the same thing as the raw column count of the engineered
        # CSV, so it can't be derived from df.shape - it only reflects
        # dataset availability.
        features = 33 if dataset_loaded else None

        hybrid_model_loaded = _check_hybrid_model_loaded()

        trend = []
        if dataset_loaded:
            trend = [
                {"year": item["year"], "crime_count": item["total_crimes"]}
                for item in _get_yearly_crime_trend()
            ]

        return JsonResponse({
            "status": "success",
            "dataset": {
                "records": records,
                "features": features,
                "period": period,
            },
            "models": {
                "cnn": {"accuracy": MODEL_PERFORMANCE_METRICS["cnn"]["accuracy"]},
                "lstm": {"accuracy": MODEL_PERFORMANCE_METRICS["lstm"]["accuracy"]},
                "hybrid": {"accuracy": MODEL_PERFORMANCE_METRICS["hybrid"]["accuracy"]},
                "hybrid_gcn": {"accuracy": MODEL_PERFORMANCE_METRICS["hybrid_gcn"]["accuracy"]},
                "hybrid_accuracy": MODEL_PERFORMANCE_METRICS["hybrid"]["accuracy"],
                "hybrid_gcn_accuracy": MODEL_PERFORMANCE_METRICS["hybrid_gcn"]["accuracy"],
            },
            "system": {
                "dataset_loaded": dataset_loaded,
                "models_loaded": hybrid_model_loaded,
                "prediction_ready": dataset_loaded,
            },
            "trend": trend,
        })
    except Exception as error:
        logger.error("api_dashboard failed: %s", error)
        return JsonResponse({"status": "error", "message": str(error)}, status=500)

def api_forecast_trend(request):
    """GET /api/forecast-trend/

    Fetches historical crime trend data for a specified state and district.
    """
    state = request.GET.get("state")
    district = request.GET.get("district")

    if not state or not district:
        return JsonResponse({
            "status": "error",
            "message": "State and district are required."
        }, status=400)

    try:
        df = dataset_loader.get_dataframe()

        if df is None or df.empty:
            return JsonResponse({
                "status": "error",
                "message": "Dataset could not be loaded."
            }, status=500)

        # Normalize column key lookup
        state_col = "STATE/UT" if "STATE/UT" in df.columns else "State/UT"
        district_col = "DISTRICT" if "DISTRICT" in df.columns else "District"
        year_col = "YEAR" if "YEAR" in df.columns else "Year"
        crime_col = "TOTAL IPC CRIMES" if "TOTAL IPC CRIMES" in df.columns else "Total Ipc Crimes"

        filtered = df[
            (df[state_col].astype(str).str.strip().str.upper() == str(state).strip().upper()) &
            (df[district_col].astype(str).str.strip().str.upper() == str(district).strip().upper())
        ].copy()

        if filtered.empty:
            return JsonResponse({
                "status": "error",
                "message": "No historical data found for this district."
            }, status=404)

        filtered[year_col] = pd.to_numeric(filtered[year_col], errors="coerce")
        filtered[crime_col] = pd.to_numeric(filtered[crime_col], errors="coerce")
        filtered = filtered.dropna(subset=[year_col, crime_col]).sort_values(year_col)

        years = [int(y) for y in filtered[year_col].tolist()]
        values = [int(v) for v in filtered[crime_col].tolist()]

        return JsonResponse({
            "status": "success",
            "years": years,
            "values": values
        })

    except Exception as e:
        return JsonResponse({
            "status": "error",
            "message": str(e)
        }, status=500)


@csrf_exempt
def api_predict(request):
    if request.method != "POST":
        return JsonResponse(
            {"status": "error", "message": "Only POST allowed"},
            status=405,
        )

    try:
        if request.content_type == "application/json":
            data = json.loads(request.body.decode("utf-8"))
        else:
            data = request.POST

        state = str(data.get("state", "")).strip()
        district = str(data.get("district", "")).strip()
        year_raw = str(data.get("year", "")).strip()

        if not state:
            return JsonResponse(
                {"status": "error", "message": "State is required."},
                status=400,
            )
        if not district:
            return JsonResponse(
                {"status": "error", "message": "District is required."},
                status=400,
            )
        if not year_raw:
            return JsonResponse(
                {"status": "error", "message": "Prediction year is required."},
                status=400,
            )

        year = int(year_raw)
        predictor = _get_crime_predictor()

        if predictor is None or not hasattr(predictor, "predict"):
            return JsonResponse(
                {
                    "status": "error",
                    "message": "Hybrid CNN-LSTM prediction service is unavailable.",
                },
                status=503,
            )

        model_type = str(data.get("model_type", "hybrid_gcn")).strip()
        result = _prediction_result_to_dict(
            predictor.predict(state, district, year, model_type=model_type)
        )

        predicted_count = result.get(
            "predicted_crime_count",
            result.get("predicted_count"),
        )

        # The Forecast page uses /api/predict/ (not /api/forecast/).
        # Ensure the response always contains the risk metadata expected
        # by forecast.js, even when the predictor returns only risk_level.
        risk_level = str(result.get("risk_level") or "Low").strip()

        risk_score = result.get("risk_score")
        try:
            if risk_score is None or str(risk_score).strip() == "" or pd.isna(risk_score):
                risk_score = None
        except (TypeError, ValueError):
            pass

        if risk_score is None:
            if risk_level == "Very High":
                risk_score = 95
            elif risk_level == "High":
                risk_score = 90
            elif risk_level in ("Moderate", "Medium"):
                risk_score = 60
            else:
                risk_score = 30

        recommendation = result.get("recommendation")
        if not recommendation:
            if risk_level == "Very High":
                recommendation = (
                    "Deploy immediate additional patrols and coordinate with local "
                    "law enforcement on targeted intervention in this district."
                )
            elif risk_level == "High":
                recommendation = (
                    "Increase police patrols and surveillance in identified hotspots."
                )
            elif risk_level in ("Moderate", "Medium"):
                recommendation = (
                    "Continuous monitoring and preventive measures are recommended."
                )
            else:
                recommendation = (
                    "Current crime trends appear stable. Continue regular monitoring."
                )

        if predicted_count is None:
            return JsonResponse(
                {
                    "status": "error",
                    "message": "Prediction service did not return a crime count.",
                },
                status=500,
            )

        # Attach real geographic neighbors for GCN Topology Card (O(1) in-memory cache)
        neighbors = []
        try:
            global _DISTRICT_NEIGHBORS_CACHE
            if _DISTRICT_NEIGHBORS_CACHE is None:
                neighbors_file = Path(settings.BASE_DIR) / "predictor" / "static" / "data" / "district_neighbors.json"
                if neighbors_file.exists():
                    with open(neighbors_file, "r", encoding="utf-8") as nf:
                        _DISTRICT_NEIGHBORS_CACHE = json.load(nf)
                else:
                    _DISTRICT_NEIGHBORS_CACHE = {}

            st_norm = str(state).strip().upper()
            dt_norm = str(district).strip().upper()
            
            STATE_ALIASES = {
                "UTTARANCHAL": "UTTARAKHAND",
                "UTTARAKHAND": "UTTARAKHAND",
                "ORISSA": "ODISHA",
                "ODISHA": "ODISHA",
                "DELHI": "DELHI UT",
                "DELHI UT": "DELHI UT",
                "PONDICHERRY": "PUDUCHERRY",
                "PUDUCHERRY": "PUDUCHERRY",
                "A & N ISLANDS": "A & N ISLANDS",
                "ANDAMAN & NICOBAR": "A & N ISLANDS",
                "ANDAMAN AND NICOBAR": "A & N ISLANDS",
                "D & N HAVELI": "D & N HAVELI",
                "DADRA & NAGAR HAVELI": "D & N HAVELI",
                "DADRA AND NAGAR HAVELI": "D & N HAVELI",
                "JAMMU AND KASHMIR": "JAMMU & KASHMIR",
                "JAMMU & KASHMIR": "JAMMU & KASHMIR",
                "TELANGANA": "ANDHRA PRADESH"
            }
            
            key = f"{st_norm}___{dt_norm}"
            if key in _DISTRICT_NEIGHBORS_CACHE:
                neighbors = _DISTRICT_NEIGHBORS_CACHE[key].get("neighbors", [])
            else:
                alias_state = STATE_ALIASES.get(st_norm, st_norm)
                alias_key = f"{alias_state}___{dt_norm}"
                if alias_key in _DISTRICT_NEIGHBORS_CACHE:
                    neighbors = _DISTRICT_NEIGHBORS_CACHE[alias_key].get("neighbors", [])
                else:
                    for k, val in _DISTRICT_NEIGHBORS_CACHE.items():
                        if k.endswith(f"___{dt_norm}"):
                            neighbors = val.get("neighbors", [])
                            break
        except Exception as e:
            logger.warning("Failed to load neighbors for %s, %s: %s", state, district, e)
            neighbors = []

        return JsonResponse(
            {
                "status": "success",
                "state": result.get("state", state),
                "district": result.get("district", district),
                "year": result.get("year", year),
                "predicted_count": round(float(predicted_count)),
                "risk_level": risk_level,
                "risk_score": round(float(risk_score), 2),
                "confidence": result.get("confidence"),
                "recommendation": recommendation,
                "model": result.get("model", "Hybrid CNN-LSTM"),
                "neighbors": neighbors,
            }
        )

    except (json.JSONDecodeError, UnicodeDecodeError):
        return JsonResponse(
            {"status": "error", "message": "Invalid request data format."},
            status=400,
        )
    except ValueError:
        return JsonResponse(
            {"status": "error", "message": "Invalid prediction year."},
            status=400,
        )
    except Exception as error:
        logger.exception("api_predict failed")
        return JsonResponse(
            {"status": "error", "message": str(error)},
            status=500,
        )


@csrf_exempt
@require_POST
def api_forecast(request):
    """POST /api/forecast/ using the existing Hybrid CNN-LSTM predictor."""
    try:
        data = json.loads(request.body.decode("utf-8"))

        state = str(data.get("state", "")).strip()
        district = str(data.get("district", "")).strip()
        year_raw = str(data.get("year", "")).strip()

        if not state or not district or not year_raw:
            return JsonResponse(
                {
                    "status": "error",
                    "message": "State, district and forecast year are required.",
                },
                status=400,
            )

        year = int(year_raw)
        predictor = _get_crime_predictor()

        if predictor is None or not hasattr(predictor, "predict"):
            return JsonResponse(
                {
                    "status": "error",
                    "message": "Hybrid CNN-LSTM prediction service is unavailable.",
                },
                status=503,
            )

        result = _prediction_result_to_dict(
            predictor.predict(state, district, year)
        )

        predicted_count = result.get(
            "predicted_crime_count",
            result.get("predicted_count"),
        )

        # Always provide complete risk metadata to the Forecast page.
        # Some predictor versions return the risk level but omit the
        # derived risk score/recommendation. Fill those fields here so
        # the UI never displays an em-dash or "No recommendation".
        risk_level = result.get("risk_level") or "Low"
        risk_score = result.get("risk_score")
        if risk_score is None:
            if risk_level == "Very High":
                risk_score = 95
            elif risk_level == "High":
                risk_score = 90
            elif risk_level in ("Moderate", "Medium"):
                risk_score = 60
            else:
                risk_score = 30

        recommendation = result.get("recommendation")
        if not recommendation:
            if risk_level == "Very High":
                recommendation = (
                    "Deploy immediate additional patrols and coordinate with local "
                    "law enforcement on targeted intervention in this district."
                )
            elif risk_level == "High":
                recommendation = (
                    "High crime risk detected. Authorities should strengthen "
                    "monitoring, patrolling and preventive measures."
                )
            elif risk_level in ("Moderate", "Medium"):
                recommendation = (
                    "Moderate crime risk detected. Continue monitoring historical "
                    "trends and maintain preventive measures."
                )
            else:
                recommendation = (
                    "Low crime risk detected. Continue regular monitoring and "
                    "standard preventive measures."
                )

        if predicted_count is None:
            return JsonResponse(
                {
                    "status": "error",
                    "message": "Prediction service did not return a crime count.",
                },
                status=500,
            )

        return JsonResponse(
            {
                "status": "success",
                "forecast": {
                    "state": result.get("state", state),
                    "district": result.get("district", district),
                    "year": result.get("year", year),
                    "predicted_count": round(float(predicted_count)),
                    "risk_level": risk_level,
                    "risk_score": risk_score,
                    "confidence": result.get("confidence"),
                    "recommendation": recommendation,
                    "model": result.get("model", "Hybrid CNN-LSTM"),
                },
                "historical_trend": _get_yearly_crime_trend_for_location(
                    state, district
                ),
            }
        )

    except (json.JSONDecodeError, UnicodeDecodeError):
        return JsonResponse(
            {"status": "error", "message": "Invalid request data format."},
            status=400,
        )
    except ValueError:
        return JsonResponse(
            {"status": "error", "message": "Invalid forecast year."},
            status=400,
        )
    except Exception as error:
        logger.exception("api_forecast failed")
        return JsonResponse(
            {"status": "error", "message": str(error)},
            status=500,
        )
