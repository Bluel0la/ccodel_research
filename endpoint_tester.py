from typing import Dict, List, Union
from google.cloud import aiplatform
from dotenv import load_dotenv
import os

load_dotenv(".env")
GCP_PROJECT = os.getenv("GCP_PROJECT")
VERTEX_ENDPOINT_ID = os.getenv("VERTEX_ENDPOINT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION")


def predict_custom_trained_model(
    project: str,
    endpoint_id: str,
    instances: Union[Dict, List[Dict]],
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
    parameters: Dict = None,  # Allow optional parameters
):
    """
    Sends a prediction request to a Vertex AI custom model.

    Args:
        project: GCP project ID.
        endpoint_id: Deployed endpoint ID.
        instances: A dictionary or list of dictionaries containing input data.
        location: Region of the model deployment.
        api_endpoint: API endpoint for the location.
        parameters: Optional parameters for model inference.

    Returns:
        Model prediction results or None if an error occurs.
    """

    # Set up API client
    client_options = {"api_endpoint": api_endpoint}
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

    # Ensure instances are in list format
    instances = instances if isinstance(instances, list) else [instances]

    # ✅ Handle missing 'text' keys safely
    formatted_instances = [
        {"inputs": instance.get("text", "")} for instance in instances
    ]

    # Default to empty dict if parameters are None
    parameters = parameters or {}

    # Construct the full endpoint path
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )

    # Send the request with error handling
    try:
        response = client.predict(
            endpoint=endpoint, instances=formatted_instances, parameters=parameters
        )
        print("Response:")
        print("Deployed Model ID:", response.deployed_model_id)

        predictions = response.predictions
        for prediction in predictions:
            print("Prediction:", prediction)

        return predictions  # Return predictions for further processing

    except Exception as e:
        print(f"Error during prediction: {e}")
        return None


# Example usage
predict_custom_trained_model(
    project=GCP_PROJECT,
    endpoint_id=VERTEX_ENDPOINT_ID,
    location=GCP_LOCATION,
    instances={"text": "Hello, how are you?"},  # ✅ Handled safely
)
