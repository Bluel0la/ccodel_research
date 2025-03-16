from typing import Dict, List, Union
from google.cloud import aiplatform

def predict_custom_trained_model(
    project: str,
    endpoint_id: str,
    instances: Union[Dict, List[Dict]],
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    """
    Sends a prediction request to a Vertex AI custom model.

    Args:
        project: GCP project ID.
        endpoint_id: Deployed endpoint ID.
        instances: A dictionary or list of dictionaries containing input data.
        location: Region of the model deployment.
        api_endpoint: API endpoint for the location.

    Returns:
        Model prediction results.
    """

    # Set up API client
    client_options = {"api_endpoint": api_endpoint}
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

    # Ensure instances are in list format
    instances = instances if isinstance(instances, list) else [instances]

    # ✅ Ensure `inputs` field is used correctly
    formatted_instances = [{"inputs": instance["text"]} for instance in instances]

    # Define parameters (optional, can be adjusted)
    parameters = {}

    # Construct the full endpoint path
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )

    # Send the request
    response = client.predict(endpoint=endpoint, instances=formatted_instances, parameters=parameters)

    # Print results
    print("Response:")
    print("Deployed Model ID:", response.deployed_model_id)

    predictions = response.predictions
    for prediction in predictions:
        print("Prediction:", prediction)

    return predictions  # Return predictions for further processing

# Example usage
predict_custom_trained_model(
    project="178740183321",
    endpoint_id="3947058727227490304",
    location="us-central1",
    instances={"text": "Hello, how are you?"}  # ✅ Ensured correct format
)
