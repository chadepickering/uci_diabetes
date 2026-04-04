# Undeploy the endpoint when not in use to avoid unnecessary costs

from google.cloud import aiplatform

aiplatform.init(project="uci-diabetes-ml", location="us-central1")
endpoint = aiplatform.Endpoint(
    "projects/616293009401/locations/us-central1/endpoints/6174031018601742336"
)
endpoint.undeploy_all()
print("Endpoint undeployed successfully")