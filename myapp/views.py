# myapp/views.py

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from tensorflow import keras
import numpy as np
from PIL import Image
import io


class PredictView(APIView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = keras.models.load_model('C:/Users/oladi/PycharmProjects/myproject/model.h5')
        self.image_size = (128, 128)  # Change to 128x128 to match the model's expected input
        self.class_names = ["Non_Autistic", "Autistic"]  # Update with actual class names if different

    def post(self, request):
        try:
            print("Received request:", request.data)

            # Get the uploaded file
            image_file = request.FILES['file']
            image = Image.open(image_file)
            print("Image loaded successfully.")

            image = image.resize(self.image_size)
            image = np.array(image) / 255.0  # Normalize the image
            print(f"Image resized to {self.image_size} and normalized.")

            # Check if image shape is correct
            if image.shape != (128, 128, 3):
                print(f"Unexpected image shape: {image.shape}")
                return Response({'error': 'Invalid image shape'}, status=status.HTTP_400_BAD_REQUEST)

            image = np.expand_dims(image, axis=0)  # Add batch dimension
            print("Image reshaped for prediction.")

            # Make prediction
            prediction = self.model.predict(image)
            predicted_probability = prediction[0][0]
            predicted_class = self.class_names[int(predicted_probability > 0.5)]
            print("Prediction made successfully:", prediction)

            return Response({
                'prediction': predicted_class,
                'confidence': float(predicted_probability),
                'all_predictions': {
                    "Non_Autistic": float(1 - predicted_probability),
                    "Autistic": float(predicted_probability)
                }
            }, status=status.HTTP_200_OK)
        except Exception as e:
            print("Error during prediction:", str(e))
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
