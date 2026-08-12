from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

API_ROOT = "https://rest.runpod.io/v1"
DEFAULT_IMAGE = "runpod/pytorch@sha256:61a4aafb0094cd773f11eefa378929d5a687bd775febeb78eac62fc824141fb5"
DEFAULT_GPU_TYPES = [
    "NVIDIA A40",
    "NVIDIA RTX A6000",
    "NVIDIA GeForce RTX 4090",
]


def _key() -> str:
    value = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not value:
        raise RuntimeError("RUNPOD_API_KEY is not set")
    return value


def api_request(method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        API_ROOT + path,
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {_key()}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
            if not raw:
                return None
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"RunPod {method} {path} failed with HTTP {exc.code}: {detail}") from exc


def hourly_cost(pod: dict[str, Any]) -> float:
    for key in ("adjustedCostPerHr", "costPerHr"):
        value = pod.get(key)
        if value not in (None, ""):
            return float(value)
    machine = pod.get("machine") or {}
    for key in ("currentPricePerGpu", "costPerHr"):
        value = machine.get(key)
        if value not in (None, ""):
            return float(value)
    raise RuntimeError("RunPod response did not include an hourly cost")


def pod_payload(
    name: str,
    public_key: str,
    cloud_type: str,
    gpu_types: list[str],
    image: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "imageName": image,
        "cloudType": cloud_type,
        "computeType": "GPU",
        "gpuCount": 1,
        "gpuTypeIds": gpu_types,
        "gpuTypePriority": "custom",
        "containerDiskInGb": 40,
        "volumeInGb": 60,
        "volumeMountPath": "/workspace",
        "ports": ["22/tcp"],
        "supportPublicIp": True,
        "interruptible": False,
        "locked": False,
        "minRAMPerGPU": 32,
        "minVCPUPerGPU": 4,
        "allowedCudaVersions": ["12.4"],
        "env": {"PUBLIC_KEY": public_key},
    }


def delete_pod(pod_id: str) -> None:
    if not pod_id:
        return
    try:
        api_request("DELETE", f"/pods/{pod_id}")
    except RuntimeError as exc:
        if "HTTP 404" not in str(exc):
            raise


def wait_for_connection(pod_id: str, timeout_seconds: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        last = api_request("GET", f"/pods/{pod_id}")
        public_ip = last.get("publicIp")
        mappings = last.get("portMappings") or {}
        ssh_port = mappings.get("22") or mappings.get(22)
        if public_ip and ssh_port:
            return last
        time.sleep(5)
    raise TimeoutError(f"Pod {pod_id} did not expose SSH in {timeout_seconds} seconds; last={last}")


def write_outputs(payload: dict[str, Any]) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    with open(output_path, "a", encoding="utf-8") as handle:
        for key in ("pod_id", "host", "port", "cost_per_hr", "gpu_type", "cloud_type"):
            handle.write(f"{key}={payload[key]}\n")


def create(args: argparse.Namespace) -> None:
    public_key = Path(args.public_key_file).read_text().strip()
    pod: dict[str, Any] | None = None
    cloud_used = ""
    errors: list[str] = []
    availability_attempts = max(
        1, int(os.environ.get("RUNPOD_AVAILABILITY_RETRIES", "12"))
    )

    for availability_attempt in range(1, availability_attempts + 1):
        errors = []
        for cloud in args.clouds:
            try:
                pod = api_request(
                    "POST",
                    "/pods",
                    pod_payload(args.name, public_key, cloud, args.gpu_types, args.image),
                )
                cloud_used = cloud
                break
            except RuntimeError as exc:
                errors.append(f"{cloud}: {exc}")

        if pod is not None:
            break

        capacity_only = bool(errors) and all(
            "no instances currently available" in error.lower() for error in errors
        )
        if capacity_only and availability_attempt < availability_attempts:
            delay_seconds = min(30, 5 + availability_attempt * 2)
            print(
                "RunPod has no requested GPU capacity; "
                f"retrying availability {availability_attempt + 1}/{availability_attempts} "
                f"in {delay_seconds}s",
                flush=True,
            )
            time.sleep(delay_seconds)
            continue

        break

    if pod is None:
        raise RuntimeError("Could not create a RunPod Pod: " + " | ".join(errors))

    pod_id = str(pod["id"])
    try:
        cost = hourly_cost(pod)
        if cost > args.max_hourly_cost:
            raise RuntimeError(
                f"Allocated Pod costs ${cost:.4f}/h, above cap ${args.max_hourly_cost:.4f}/h"
            )
        live = wait_for_connection(pod_id, args.wait_seconds)
        gpu = live.get("gpu") or {}
        machine = live.get("machine") or {}
        gpu_type = gpu.get("displayName") or machine.get("gpuDisplayName") or machine.get("gpuTypeId") or "unknown"
        mappings = live.get("portMappings") or {}
        result = {
            "pod_id": pod_id,
            "host": str(live["publicIp"]),
            "port": int(mappings.get("22") or mappings.get(22)),
            "cost_per_hr": cost,
            "gpu_type": gpu_type,
            "cloud_type": cloud_used,
            "image": args.image,
            "raw_create": pod,
            "raw_live": live,
        }
        Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        write_outputs(result)
        print(
            f"Allocated {gpu_type} in {cloud_used} at ${cost:.4f}/h; "
            f"pod={pod_id} ssh={result['host']}:{result['port']}"
        )
    except Exception:
        delete_pod(pod_id)
        raise


def auth_check(_: argparse.Namespace) -> None:
    pods = api_request("GET", "/pods")
    if not isinstance(pods, list):
        raise RuntimeError("Unexpected RunPod authentication response")
    print(f"RunPod API authentication succeeded; visible pods={len(pods)}")


def delete(args: argparse.Namespace) -> None:
    delete_pod(args.pod_id)
    print(f"Terminated RunPod Pod {args.pod_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Guarded RunPod lifecycle operations for release experiments")
    sub = parser.add_subparsers(dest="command", required=True)

    auth = sub.add_parser("auth-check")
    auth.set_defaults(func=auth_check)

    create_parser = sub.add_parser("create")
    create_parser.add_argument("--name", required=True)
    create_parser.add_argument("--public-key-file", required=True)
    create_parser.add_argument("--max-hourly-cost", type=float, default=0.75)
    create_parser.add_argument("--wait-seconds", type=int, default=900)
    create_parser.add_argument("--clouds", nargs="+", default=["SECURE", "COMMUNITY"])
    create_parser.add_argument("--gpu-types", nargs="+", default=DEFAULT_GPU_TYPES)
    create_parser.add_argument("--image", default=DEFAULT_IMAGE)
    create_parser.add_argument("--output", default="runpod-allocation.json")
    create_parser.set_defaults(func=create)

    delete_parser = sub.add_parser("delete")
    delete_parser.add_argument("--pod-id", required=True)
    delete_parser.set_defaults(func=delete)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
