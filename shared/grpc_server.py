"""Serves a backend's servicer on the gRPC address LocalAI passes in `--addr`."""

import argparse
import signal
import sys
from concurrent import futures

import backend_pb2_grpc
import grpc
from grpc_auth import get_auth_interceptors


def run(servicer: backend_pb2_grpc.BackendServicer, name: str) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--addr", default="localhost:50051")
    address = parser.parse_args().addr
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=1),
        options=[("grpc.max_receive_message_length", 50 * 1024 * 1024)],
        interceptors=get_auth_interceptors(),
    )
    backend_pb2_grpc.add_BackendServicer_to_server(servicer, server)
    server.add_insecure_port(address)
    server.start()
    print(f"[{name}] listening on {address}", file=sys.stderr)

    def stop(_sig, _frame):
        server.stop(0)
        sys.exit(0)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    server.wait_for_termination()
