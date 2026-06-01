from __future__ import annotations

import argparse

import boto3

from config.settings import settings
from src.data import DataLoader
from src.data.data_sources import DataSourceManager


def endpoint_url() -> str:
    endpoint = settings.minio_endpoint
    if endpoint.startswith("http://") or endpoint.startswith("https://"):
        return endpoint
    return f"http://{endpoint}"


def ensure_bucket(bucket_name: str) -> None:
    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url(),
        aws_access_key_id=settings.minio_access_key,
        aws_secret_access_key=settings.minio_secret_key,
    )
    existing = {bucket["Name"] for bucket in client.list_buckets().get("Buckets", [])}
    if bucket_name not in existing:
        client.create_bucket(Bucket=bucket_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload configured dataset from CSV/local source to S3/MinIO")
    parser.add_argument("--dataset", required=True, help="base | variant_1 | variant_2 | samld")
    parser.add_argument("--from-source", default="csv", help="Source to read from, default: csv")
    parser.add_argument("--to-source", default="s3", help="Destination source, default: s3")
    parser.add_argument("--nrows", type=int, default=None, help="Optional row limit for test uploads")
    args = parser.parse_args()

    manager = DataSourceManager()
    target = manager.resolve(args.dataset, source_override=args.to_source)
    uri = target.config.get("uri", "")
    if uri.startswith("s3://"):
        bucket = uri.replace("s3://", "", 1).split("/", 1)[0]
        ensure_bucket(bucket)

    loader = DataLoader()
    read_options = {"nrows": args.nrows} if args.nrows else {}
    df = loader.load_dataset(args.dataset, source_override=args.from_source, **read_options)
    loader.save_dataset_to_source(df, args.dataset, source_override=args.to_source, index=False)

    print(f"OK: uploaded dataset={args.dataset}, rows={len(df)}, destination={args.to_source}")


if __name__ == "__main__":
    main()
