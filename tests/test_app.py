from pathlib import Path

from app import create_app


def test_index_page():
    app = create_app()
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"Prediksi Indikasi Penyakit Jantung" in response.data


def test_predict_valid_input():
    app = create_app()
    client = app.test_client()
    response = client.post("/predict", data={
        "age": "52",
        "sex": "1",
        "cp": "3",
        "trestbps": "125",
        "chol": "212",
        "fbs": "0",
        "restecg": "1",
        "thalach": "168",
        "exang": "0",
        "oldpeak": "1.0",
        "slope": "2",
        "ca": "0",
        "thal": "3",
    })
    assert response.status_code == 200
    assert b"Probabilitas" in response.data


def test_predict_invalid_input():
    app = create_app()
    client = app.test_client()
    response = client.post("/predict", data={"age": "", "sex": "2"})
    assert response.status_code == 200
    assert b"Wajib diisi" in response.data
