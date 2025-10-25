import os
import json
import base64
import boto3


def lambda_handler(event, context):
    ENDPOINT = os.getenv("ENDPOINT")
    if not ENDPOINT:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "ENDPOINT env var not set"}),
            "headers": {"Access-Control-Allow-Origin": "*"},
        }
    body = event.get("body")
    if isinstance(body, str):
        body = json.loads(body)
    audio_b64 = body.get("audio_b64")
    if not audio_b64:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Missing audio_b64"}),
            "headers": {"Access-Control-Allow-Origin": "*"},
        }
    audio_bytes = base64.b64decode(audio_b64)
    client = boto3.client("sagemaker-runtime")
    response = client.invoke_endpoint(
        EndpointName=ENDPOINT, ContentType="audio/wav", Body=audio_bytes
    )
    result = json.loads(response["Body"].read())
    return {
        "statusCode": 200,
        "body": json.dumps(result),
        "headers": {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
        },
    }
