# generate.py — CPU-parallel rainbow table builder
#
# Architecture:
#   - No GPU. Targets the high-memory CPU partition (96 cores, 1495 GB RAM).
#   - SHA-256 + MurmurHash3 are implemented in C via cffi and compiled once
#     at startup.  Each call hashes one uint64 point and reduces it mod N —
#     identical semantics to the original Python sha256/mmh3 calls.
#   - The column lives in a SharedMemory block.  All worker processes attach
#     to it by name — zero copies across process boundaries.
#   - Two parallelism levels:
#       1. Hash-reduce phase: column is split into CORES chunks, each worker
#          hashes its chunk and writes results into a second shared output
#          block.  Workers run concurrently, then main assembles the output.
#       2. Seed-search phase: seeds are split across CORES workers, each
#          worker reads the full (already-reduced) column from shared memory
#          and counts uniques for its batch of seeds.
#   - Unique counting uses numpy on each worker's output slice — with 1.5 TB
#     of host RAM there is no memory pressure.
#   - The Pool is created once before the build loop so there is no
#     per-column spawn or compile overhead.

import ctypes
# import tempfile
# import subprocess
from math import ceil
from multiprocessing import shared_memory
import multiprocessing as mp
import os
from time import perf_counter
import random
import numpy as np


###############################################################################
# Constants
###############################################################################

N     = 2 ** 16
alpha = 0.95
t     = 80
CORES = 16          # physical cores on the high-mem node

def calculate_m0(N, t, alpha):
    mttarget = (2 * N) / (t + 2)
    return (alpha * mttarget) / (1 - alpha)

m0 = calculate_m0(N, t, alpha)

filename = "N40_kcurve.pkl"
kis = np.load(filename, allow_pickle=True)


###############################################################################
# C implementation of SHA-256 + MurmurHash3
#
# SHA-256: uses OpenSSL's EVP API (libcrypto), which is present on every
#   modern Linux HPC cluster as a system package.  OpenSSL uses hardware
#   SHA-NI instructions automatically when available on the CPU — typically
#   4-6x faster than a software implementation.
#   API used: EVP_MD_CTX (OpenSSL >= 1.1.0, which covers all current systems).
#
# MurmurHash3: aappleby/smhasher reference implementation, public domain.
#   ~40 lines, no dependencies, identical output to Python's mmh3 package.
#   https://github.com/aappleby/smhasher
#
# The exported hash_reduce_array() processes an entire uint64 array in a
# tight C loop — no Python overhead per element.
###############################################################################

_C_SOURCE = r"""
#include <stdint.h>
#include <string.h>
#include <openssl/evp.h>    // OpenSSL EVP API for SHA-256

// ── MurmurHash3_x86_32  —  aappleby/smhasher (public domain) ────────────────
// Verbatim from Austin Appleby's reference implementation.
// Produces identical output to Python: mmh3.hash(data, seed, signed=False)

#define ROTL32(x,y) (((x)<<(y))|((x)>>(32-(y))))

static uint32_t fmix32(uint32_t h)
{
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

static uint32_t MurmurHash3_x86_32(const void *key, int len, uint32_t seed)
{
    const uint8_t  *data    = (const uint8_t *)key;
    const int       nblocks = len / 4;
    uint32_t h1 = seed;
    const uint32_t c1 = 0xcc9e2d51u, c2 = 0x1b873593u;

    const uint32_t *blocks = (const uint32_t *)(data + nblocks * 4);
    for (int i = -nblocks; i; i++) {
        uint32_t k1 = blocks[i];
        k1 *= c1; k1 = ROTL32(k1, 15); k1 *= c2;
        h1 ^= k1; h1 = ROTL32(h1, 13); h1 = h1 * 5 + 0xe6546b64u;
    }

    const uint8_t *tail = data + nblocks * 4;
    uint32_t k1 = 0;
    switch (len & 3) {
        case 3: k1 ^= (uint32_t)tail[2] << 16; /* fall through */
        case 2: k1 ^= (uint32_t)tail[1] << 8;  /* fall through */
        case 1: k1 ^= (uint32_t)tail[0];
                k1 *= c1; k1 = ROTL32(k1, 15); k1 *= c2; h1 ^= k1;
    }
    h1 ^= len;
    return fmix32(h1);
}


// ── Exported: hash-reduce a contiguous block of uint64 points ───────────────
//
// For each point x:
//   digest = SHA-256(x.to_bytes(8, 'little'))   via OpenSSL EVP
//   result = MurmurHash3_x86_32(digest, 32, seed) % N_val
//
// input  : uint64 array of n points
// seed   : encodes column index + rf sub-index
// N_val  : universe size
// output : caller-allocated uint64 array of n results

void hash_reduce_array(
    const uint64_t *input,
    uint64_t        n,
    uint32_t        seed,
    uint64_t        N_val,
    uint64_t       *output)
{
    // Allocate one EVP context and reuse it for every point.
    // EVP_MD_CTX_reset() reinitialises without reallocation — much cheaper
    // than creating/destroying a context per element.
    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    const EVP_MD *md = EVP_sha256();

    for (uint64_t idx = 0; idx < n; ++idx) {
        // Step 1: 8-byte little-endian representation of the point
        uint64_t val = input[idx];
        uint8_t  msg[8];
        msg[0] = (uint8_t)( val        & 0xff);
        msg[1] = (uint8_t)((val >>  8) & 0xff);
        msg[2] = (uint8_t)((val >> 16) & 0xff);
        msg[3] = (uint8_t)((val >> 24) & 0xff);
        msg[4] = (uint8_t)((val >> 32) & 0xff);
        msg[5] = (uint8_t)((val >> 40) & 0xff);
        msg[6] = (uint8_t)((val >> 48) & 0xff);
        msg[7] = (uint8_t)((val >> 56) & 0xff);

        // Step 2: SHA-256 via OpenSSL EVP — uses SHA-NI hardware if available
        uint8_t  digest[32];
        unsigned int digest_len = 32;
        EVP_DigestInit_ex(ctx, md, NULL);
        EVP_DigestUpdate(ctx, msg, 8);
        EVP_DigestFinal_ex(ctx, digest, &digest_len);

        // Step 3: MurmurHash3 over the 32-byte digest, reduce mod N
        uint32_t h = MurmurHash3_x86_32(digest, 32, seed);
        output[idx] = (uint64_t)h % N_val;
    }

    EVP_MD_CTX_free(ctx);
}
"""

# ── Compile the C source to a shared library at startup ─────────────────────

def _compile_hash_lib() -> str:
    """
    Compile _C_SOURCE against OpenSSL libcrypto and return the .so path.
    Requires: gcc, openssl-devel (or libssl-dev) — standard on HPC clusters.
    If OpenSSL headers aren't in the default include path, set OPENSSL_PREFIX
    as an environment variable, e.g.:
        export OPENSSL_PREFIX=/usr   (default, works on most systems)
        export OPENSSL_PREFIX=$EBROOTOPENSSL   (EasyBuild clusters)
    """
    import os, tempfile, subprocess

    openssl_prefix = os.environ.get('OPENSSL_PREFIX', '/usr')
    include_flag   = f'-I{openssl_prefix}/include'
    lib_flag       = f'-L{openssl_prefix}/lib64'   # lib64 on RHEL/Rocky/CentOS

    src = tempfile.NamedTemporaryFile(suffix='.c', delete=False, mode='w')
    src.write(_C_SOURCE)
    src.close()

    lib_path = src.name.replace('.c', '.so')
    subprocess.check_call([
        'gcc', '-O3', '-march=native', '-shared', '-fPIC',
        include_flag, lib_flag,
        '-o', lib_path, src.name,
        '-lcrypto',          # link OpenSSL libcrypto
    ])
    os.unlink(src.name)
    return lib_path

_LIB_PATH = _compile_hash_lib()

def _load_lib() -> ctypes.CDLL:
    lib = ctypes.CDLL(_LIB_PATH)
    lib.hash_reduce_array.restype  = None
    lib.hash_reduce_array.argtypes = [
        ctypes.c_void_p,   # input  (uint64*)
        ctypes.c_uint64,   # n
        ctypes.c_uint32,   # seed
        ctypes.c_uint64,   # N_val
        ctypes.c_void_p,   # output (uint64*)
    ]
    return lib

# Load in main process (used for the final apply step)
_lib = _load_lib()


def hash_reduce_numpy(col: np.ndarray, seed: int, N_val: int) -> np.ndarray:
    """Hash-reduce a numpy uint64 array using the compiled C library."""
    out = np.empty(len(col), dtype=np.uint64)
    _lib.hash_reduce_array(
        col.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_uint64(len(col)),
        ctypes.c_uint32(seed),
        ctypes.c_uint64(N_val),
        out.ctypes.data_as(ctypes.c_void_p),
    )
    return out


###############################################################################
# Worker functions
###############################################################################

# Per-worker global — holds the loaded C library handle
_worker_lib = None


def _worker_init(lib_path: str):
    """Called once per worker at pool creation. Loads the compiled C lib."""
    global _worker_lib
    import ctypes
    _worker_lib = ctypes.CDLL(lib_path)
    _worker_lib.hash_reduce_array.restype  = None
    _worker_lib.hash_reduce_array.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_uint32,
        ctypes.c_uint64,
        ctypes.c_void_p,
    ]


# ── Phase 1: hash-reduce a chunk of the column ──────────────────────────────

def _hash_chunk(
    in_shm_name:  str,
    out_shm_name: str,
    start:        int,
    length:       int,
    seed:         int,
    N_val:        int,
):
    """
    Worker task for the hash-reduce phase.
    Reads input[start : start+length] from shared memory,
    writes hash_reduce results into output[start : start+length].
    No data is sent through the pipe — only integers.
    """
    from multiprocessing import shared_memory
    import numpy as np
    import ctypes

    in_shm  = shared_memory.SharedMemory(name=in_shm_name)
    out_shm = shared_memory.SharedMemory(name=out_shm_name)

    # Wrap shared memory as numpy arrays (zero copy)
    in_arr  = np.ndarray((start + length,), dtype=np.uint64, buffer=in_shm.buf)
    out_arr = np.ndarray((start + length,), dtype=np.uint64, buffer=out_shm.buf)

    chunk_in  = in_arr[start : start + length]
    chunk_out = out_arr[start : start + length]

    _worker_lib.hash_reduce_array(
        chunk_in.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_uint64(length),
        ctypes.c_uint32(seed),
        ctypes.c_uint64(N_val),
        chunk_out.ctypes.data_as(ctypes.c_void_p),
    )

    in_shm.close()
    out_shm.close()


# ── Phase 2: count uniques for a batch of seeds ─────────────────────────────

def _count_seeds(
    col_shm_name: str,
    n_points:     int,
    seed_batch:   np.ndarray,
    N_val:        int,
):
    """
    Worker task for the seed-search phase.
    For each seed in seed_batch, hash-reduces the full column and counts
    unique output values.  Returns (best_seed, best_count).

    The column is read from shared memory (zero copy).
    Each reduced output is allocated fresh, counted, then freed.
    """
    if len(seed_batch) == 0:
        return None, -1

    from multiprocessing import shared_memory
    import numpy as np
    import ctypes

    shm = shared_memory.SharedMemory(name=col_shm_name)
    col = np.ndarray((n_points,), dtype=np.uint64, buffer=shm.buf)

    best_count = -1
    best_seed  = int(seed_batch[0])

    for s in seed_batch:
        s_int  = int(s)
        output = np.empty(n_points, dtype=np.uint64)

        _worker_lib.hash_reduce_array(
            col.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_uint64(n_points),
            ctypes.c_uint32(s_int),
            ctypes.c_uint64(N_val),
            output.ctypes.data_as(ctypes.c_void_p),
        )

        # np.unique on a pre-sorted array is O(n) — sort first
        # output.sort()
        # # count uniques via adjacent diff (avoids np.unique's internal copy)
        # count = int(np.count_nonzero(np.diff(output))) + 1
        # del output
        count = np.unique(output).size
        del output

        if count > best_count:
            best_count = count
            best_seed  = s_int

    shm.close()
    return best_seed, best_count


###############################################################################
# Parallel hash-reduce (Phase 1 driver)
###############################################################################

def parallel_hash_reduce(
    pool:         mp.Pool,
    col_shm:      shared_memory.SharedMemory,
    out_shm:      shared_memory.SharedMemory,
    n_points:     int,
    seed:         int,
    N_val:        int,
    n_workers:    int,
) -> np.ndarray:
    """
    Hash-reduce `current_column` in parallel across n_workers cores.
    Each worker processes a contiguous chunk, writing into out_shm.
    Returns the assembled output as a numpy array.
    """
    chunk_size = ceil(n_points / n_workers)
    args = []
    for w in range(n_workers):
        start  = w * chunk_size
        length = min(chunk_size, n_points - start)
        if length <= 0:
            break
        args.append((col_shm.name, out_shm.name, start, length, seed, N_val))

    pool.starmap(_hash_chunk, args)

    # Assemble from shared memory — zero copy view
    result = np.ndarray((n_points,), dtype=np.uint64, buffer=out_shm.buf)
    return result.copy()   # copy so we own the memory independently of shm


###############################################################################
# Parallel seed search (Phase 2 driver)
###############################################################################

def parallel_seed_search(
    pool:         mp.Pool,
    col_shm:      shared_memory.SharedMemory,
    n_points:     int,
    index_sample: np.ndarray,
    N_val:        int,
    n_workers:    int,
) -> int:
    """
    Split index_sample across n_workers cores.
    Each worker counts uniques for its batch of seeds.
    Returns the seed that produced the most unique output points.
    """
    n_workers = min(n_workers, len(index_sample))
    batches   = np.array_split(index_sample, n_workers)
    args      = [(col_shm.name, n_points, b, N_val) for b in batches]

    results  = pool.starmap(_count_seeds, args)
    results  = [(s, c) for s, c in results if s is not None]
    best_seed, _ = max(results, key=lambda x: x[1])
    return best_seed


###############################################################################
# Rainbow table builder
###############################################################################

def build(m0, t, kis):
    start_time = perf_counter()

    t_bits      = (t - 1).bit_length()
    k_bits      = 32 - t_bits
    max_k_index = 2 ** k_bits

    rf_indexes     = np.zeros(t, dtype=np.uint32)
    current_column = np.arange(round(m0), dtype=np.uint64)
    startpoints    = np.arange(round(m0), dtype=np.uint64)

    # ── Shared memory layout ─────────────────────────────────────────────────
    # col_shm  : holds current_column (read by seed-search workers)
    # out_shm  : scratch space for hash-reduce output (written by hash workers)
    # Both are sized for the initial (maximum) column length.
    # As the column shrinks we just use a prefix of each block.
    col_shm = shared_memory.SharedMemory(create=True, size=current_column.nbytes)
    out_shm = shared_memory.SharedMemory(create=True, size=current_column.nbytes)

    col_shm_arr = np.ndarray(current_column.shape, dtype=np.uint64, buffer=col_shm.buf)

    # ── Create the worker pool once ──────────────────────────────────────────
    # Workers load the C library in their initialiser and stay alive for all
    # t columns — no per-column spawn or compile overhead.
    ctx = mp.get_context('spawn')

    try:
        with ctx.Pool(
            processes=CORES,
            initializer=_worker_init,
            initargs=(_LIB_PATH,),
        ) as pool:

            for i in range(t):
                k_i          = round(kis[i])
                picks        = np.array(
                    random.sample(range(max_k_index), k_i), dtype=np.uint32
                )
                index_sample = (np.uint32(i) << np.uint32(k_bits)) | picks

                n_points = len(current_column)

                # ── Phase 1: find best seed ──────────────────────────────────
                # Write current column into col_shm so seed-search workers
                # can read it without any data copying through pipes.
                np.copyto(col_shm_arr[:n_points], current_column)

                best_index = parallel_seed_search(
                    pool, col_shm, n_points, index_sample, N, CORES
                )
                rf_indexes[i] = best_index

                # ── Phase 2: apply best reduction function ───────────────────
                # Parallelise the hash-reduce of the full column across cores.
                # Workers read col_shm and write into out_shm concurrently.
                reduced = parallel_hash_reduce(
                    pool, col_shm, out_shm, n_points, int(best_index), N, CORES
                )

                # Find unique endpoints and their corresponding startpoints
                unique_pts, sp_idx = np.unique(reduced, return_index=True)
                current_column = unique_pts
                startpoints    = startpoints[sp_idx]

                del reduced, unique_pts, sp_idx

    finally:
        col_shm.close(); col_shm.unlink()
        out_shm.close(); out_shm.unlink()

    duration = perf_counter() - start_time
    print(f"\nRainbow table built in {duration:.2f} seconds")
    return startpoints, current_column, rf_indexes


###############################################################################

if __name__ == '__main__':
    startpoints, final_column, rf_indexes = build(m0, t, kis)
    
    # Save results
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    np.save(f"{output_dir}/startpoints.npy",  startpoints)
    np.save(f"{output_dir}/final_column.npy", final_column)
    np.save(f"{output_dir}/rf_indexes.npy",   rf_indexes)

    print(f"\nResults saved to {output_dir}/")
    print(f"\nFinal column length          : {len(final_column):,}")
    print(f"Startpoints / endpoints match: {len(startpoints) == len(final_column)}")
    print(f"Endpoints are unique         : {len(final_column) == len(set(final_column))}")
    print(f"Distinct rf_indexes          : {len(set(rf_indexes))} / {len(rf_indexes)}")
