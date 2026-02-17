#!/usr/bin/env python3
\"\"\"SuperPOD NVSwitch AllReduce Latency Simulator.
Simulates ring/tree comm in DGX H100 SuperPOD (8 GPUs/node).\"\"\"

import argparse
import multiprocessing as mp
import numpy as np
import time

# H100 DGX SuperPOD specs
LATENCIES_NS = {
    'nvlink': 50,   # NVSwitch hop
    'ib': 1000,     # NIC RDMA
    'pcie': 600,    # PCIe Gen5
}

BWS_GBPS = {
    'nvlink': 900,
    'ib': 400,
    'pcie': 64,
}

GPUS_PER_NODE = 8

def ring_allreduce(rank, size, payload_bytes, latency_ns, bw_gbps):
    \"\"\"Ring AllReduce sim.\"\"\"
    hops = size
    lat = hops * latency_ns * 1e-9
    bw_time = (payload_bytes * 2 * hops) / (bw_gbps * 1e9)  # Reduce-scatter + All-gather
    return lat + bw_time

def tree_allreduce(rank, size, payload_bytes, latency_ns, bw_gbps):
    \"\"\"Tree (butterfly) AllReduce.\"\"\"
    hops = np.log2(size) * 2  # Fan-in + fan-out
    lat = hops * latency_ns * 1e-9
    bw_time = payload_bytes * hops / (bw_gbps * 1e9)
    return lat + bw_time

def benchmark(transport, nodes, payload_gb):
    \"\"\"Benchmark round-trip.\"\"\"
    gpus = nodes * GPUS_PER_NODE
    payload_bytes = payload_gb * 1e9
    latency_ns = LATENCIES_NS[transport]
    bw_gbps = BWS_GBPS[transport]
    
    start = time.time()
    with mp.Pool(gpus) as pool:
        args = [(i, gpus, payload_bytes, latency_ns, bw_gbps) for i in range(gpus)]
        latencies = pool.starmap(transport + '_allreduce', args)
    end = time.time()
    
    avg_lat_us = np.mean(latencies) * 1e6
    tp_gbs = (gpus * payload_gb * 2) / (end - start)  # Bidirectional
    return avg_lat_us, tp_gbs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=int, default=4)
    parser.add_argument('--payload-gb', type=float, default=2.0)
    args = parser.parse_args()
    
    print(f"SuperPOD: {args.nodes} nodes ({args.nodes*GPUS_PER_NODE} H100s), {args.payload_gb}GB payload")
    
    for transport in LATENCIES_NS:
        lat_us, tp_gbs = benchmark(transport, args.nodes, args.payload_gb)
        print(f"{transport.upper()}: {lat_us:.1f}us lat, {tp_gbs:.1f} GB/s TP")
