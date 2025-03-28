import requests

url = 'http://localhost:5000/predict'
files = {'file': open('test_digit.png', 'rb')}
response = requests.post(url, files=files)
print(response.json())