POST /predict  

Example input:
{
  "temp_max": 20,
  "temp_min": 10,
  "precipitation": 2
}

Example output:
{
  "prediction": 0,
  "result": "No rain tomorrow"
}

---

## Monitoring

All predictions are logged in logs.log.
