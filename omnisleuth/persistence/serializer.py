"""Centralized JSON-safe serialization utilities for persistence and data transfer."""

from datetime import date, datetime, time
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import uuid
import numpy as np
import pandas as pd
from pydantic import BaseModel


def to_json_safe(obj: Any) -> Any:
    """Recursively converts an arbitrary Python / Pandas / Numpy object into JSON-serializable primitives.

    Handles:
    - pandas Timestamp -> ISO-8601 string
    - pandas NaT -> None
    - datetime, date, time -> ISO-8601 string
    - numpy integer types -> standard int
    - numpy floating types -> standard float (or None if NaN/inf)
    - numpy bool_ -> standard bool
    - numpy ndarray -> list of converted elements
    - pandas Series -> list of converted elements
    - pandas DataFrame -> list of dict records
    - pandas NA / None -> None
    - math NaN / Inf -> None
    - UUID -> string
    - Enum -> enum.value
    - Path -> string
    - Pydantic BaseModel -> recursively converted model dict
    - Dictionaries, Lists, Tuples, Sets -> recursively converted containers
    """
    if obj is None:
        return None

    # 1. Pandas missing/NaT values
    if obj is pd.NA:
        return None
    try:
        if pd.isna(obj):
            # Check if it's a single scalar NA/NaN/NaT (not a collection)
            if not isinstance(obj, (pd.DataFrame, pd.Series, np.ndarray, list, dict)):
                return None
    except (TypeError, ValueError):
        pass

    # 2. Datetime / Timestamps
    if isinstance(obj, pd.Timestamp):
        if pd.isna(obj):
            return None
        return obj.isoformat()

    if isinstance(obj, (datetime, date, time)):
        return obj.isoformat()

    # 3. Numpy numeric & boolean types
    if isinstance(obj, (np.integer, int)):
        return int(obj)

    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val

    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)

    # 4. Enums and UUIDs and Paths
    if isinstance(obj, Enum):
        return to_json_safe(obj.value)

    if isinstance(obj, (uuid.UUID, Path)):
        return str(obj)

    # 5. Pydantic models
    if isinstance(obj, BaseModel):
        return to_json_safe(obj.model_dump())

    # 6. Pandas & Numpy containers
    if isinstance(obj, pd.DataFrame):
        return [to_json_safe(row) for row in obj.to_dict(orient="records")]

    if isinstance(obj, pd.Series):
        return [to_json_safe(val) for val in obj.tolist()]

    if isinstance(obj, np.ndarray):
        return [to_json_safe(val) for val in obj.tolist()]

    # 7. Standard collections
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [to_json_safe(item) for item in obj]

    # 8. String and other primitives
    if isinstance(obj, str):
        return obj

    # Fallback to string representation for complex unknown types
    return str(obj)


def json_default_handler(obj: Any) -> Any:
    """Fallback handler for json.dumps default parameter."""
    return to_json_safe(obj)


def safe_json_dumps(obj: Any, **kwargs: Any) -> str:
    """Serialize an object to a JSON formatted string safely without raising serialization errors."""
    safe_obj = to_json_safe(obj)
    return json.dumps(safe_obj, default=json_default_handler, **kwargs)


def safe_json_loads(s: Optional[str], default: Any = None) -> Any:
    """Safely parse a JSON string into Python primitives, returning default on error or None."""
    if s is None or not str(s).strip():
        return default
    try:
        return json.loads(s)
    except Exception:
        return default
