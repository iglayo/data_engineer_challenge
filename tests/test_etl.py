import os
from datetime import datetime, timedelta
import pytest
from src.etl import fetch_esios

@pytest.mark.skipif(not os.getenv("ESIOS_API_KEY"), reason="No ESIOS_API_KEY set")
def test_fetch_esios_shape():
    end = datetime.utcnow()
    start = end - timedelta(hours=24)
    df = fetch_esios(1293, start, end)
    assert "datetime" in df.columns
    assert "target" in df.columns
    assert not df.empty