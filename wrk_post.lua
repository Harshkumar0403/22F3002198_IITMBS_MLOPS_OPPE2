wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"

-- one sample from your random data (numeric gender)
wrk.body = [[
{
  "sno": 1,
  "age": 52,
  "gender": 1,
  "cp": 1,
  "trestbps": 140,
  "chol": 240,
  "fbs": 0,
  "restecg": 1,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 1.5,
  "slope": 1,
  "ca": 0,
  "thal": 2
}
]]

