from django.shortcuts import render
import pickle
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, 'model.pkl')
vectorizer_path = os.path.join(BASE_DIR, 'vectorizer.pkl')

model = pickle.load(open(model_path, 'rb'))
vectorizer = pickle.load(open(vectorizer_path, 'rb'))

def home(request):
    prediction = None
    if request.method == 'POST':
        message = request.POST.get('message')
        transformed = vectorizer.transform([message])
        result = model.predict(transformed)[0]
        prediction = "Spam " if result == 1 else "Ham "
    return render(request, 'home.html', {'prediction': prediction})
