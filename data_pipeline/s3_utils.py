"""
S3 utilities for Acoustic Shield data pipeline.
Provides region-agnostic S3 operations.
"""
import json
import logging
from typing import Any, Dict, List, Optional
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class S3Client:
    """Wrapper for S3 operations with automatic region detection."""
    
    def __init__(self):
        """Initialize S3 client with automatic region detection."""
        self.s3_client = boto3.client('s3')
        self.s3_resource = boto3.resource('s3')
    
    def get_bucket_region(self, bucket_name: str) -> str:
        """Get the region of an S3 bucket."""
        try:
            response = self.s3_client.get_bucket_location(Bucket=bucket_name)
            region = response.get('LocationConstraint')
            # us-east-1 returns None
            return region if region else 'us-east-1'
        except ClientError as e:
            logger.error(f"Error getting bucket region: {e}")
            raise
    
    def read_json(self, bucket: str, key: str) -> Dict[str, Any]:
        """Read JSON file from S3."""
        try:
            obj = self.s3_client.get_object(Bucket=bucket, Key=key)
            content = obj['Body'].read().decode('utf-8')
            return json.loads(content)
        except ClientError as e:
            logger.error(f"Error reading s3://{bucket}/{key}: {e}")
            raise
    
    def write_json(self, data: Any, bucket: str, key: str) -> str:
        """Write JSON data to S3."""
        try:
            json_str = json.dumps(data, indent=2)
            self.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=json_str.encode('utf-8'),
                ContentType='application/json'
            )
            return f"s3://{bucket}/{key}"
        except ClientError as e:
            logger.error(f"Error writing to s3://{bucket}/{key}: {e}")
            raise
    
    def list_objects(self, bucket: str, prefix: str = '', max_keys: int = 1000) -> List[str]:
        """List objects in S3 bucket with given prefix."""
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket, Prefix=prefix, PaginationConfig={'MaxItems': max_keys})
            
            keys = []
            for page in pages:
                if 'Contents' in page:
                    keys.extend([obj['Key'] for obj in page['Contents']])
            return keys
        except ClientError as e:
            logger.error(f"Error listing objects in s3://{bucket}/{prefix}: {e}")
            raise
    
    def upload_file(self, file_path: str, bucket: str, key: str) -> str:
        """Upload a local file to S3."""
        try:
            self.s3_client.upload_file(file_path, bucket, key)
            return f"s3://{bucket}/{key}"
        except ClientError as e:
            logger.error(f"Error uploading {file_path} to s3://{bucket}/{key}: {e}")
            raise
    
    def download_file(self, bucket: str, key: str, local_path: str) -> str:
        """Download S3 object to local file."""
        try:
            self.s3_client.download_file(bucket, key, local_path)
            return local_path
        except ClientError as e:
            logger.error(f"Error downloading s3://{bucket}/{key}: {e}")
            raise
