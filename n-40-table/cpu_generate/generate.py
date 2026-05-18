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
N     = 2 ** 40
alpha = 0.95
t     = 10000
CORES = 96          # physical cores on the high-mem node

def calculate_m0(N, t, alpha):
    mttarget = (2 * N) / (t + 2)
    return (alpha * mttarget) / (1 - alpha)

m0 = calculate_m0(N, t, alpha)

filename = "N40_kcurve.pkl"
kis = np.load(filename, allow_pickle=True)


###############################################################################
# C implementation of SHA-256 + MurmurHash3

_C_SOURCE = r"""
#include <stdint.h>
#include <string.h>
#include <openssl/evp.h>    // use OpenSSL for SHA256

// MurmurHash3_x86_32  —  aappleby/smhasher (public domain)
// copied from https://github.com/aappleby/smhasher/src/MurmurHash3.cpp
// produces identical output to Python: mmh3.hash(data, seed, signed=False)

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

// end code for MurmurHash3_x86_32


// hash-reduce a contiguous block of uint64 points
//
// For each point x:
//   digest = SHA-256(x.to_bytes(8, 'little'))   via OpenSSL SHA256
//   result = MurmurHash3_x86_32(digest, 32, seed) % N_val
//
// input : uint64 array of n points
// seed : encodes column index + rf sub-index
// N_val : universe size
// output : caller-allocated uint64 array of n results

void hash_reduce_array(
    const uint64_t *input,
    uint64_t        n,
    uint32_t        seed,
    uint64_t        N_val,
    uint64_t       *output)
{
    // Allocate one OpenSSL EVP context and reuse it for every point.
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

        // SHA-256 via OpenSSL EVP — uses SHA-NI hardware if available
        uint8_t  digest[32];
        unsigned int digest_len = 32;
        EVP_DigestInit_ex(ctx, md, NULL);
        EVP_DigestUpdate(ctx, msg, 8);
        EVP_DigestFinal_ex(ctx, digest, &digest_len);

        // MurmurHash3 over the 32-byte digest, reduce mod N
        uint32_t h = MurmurHash3_x86_32(digest, 32, seed);
        output[idx] = (uint64_t)h % N_val;
    }

    // free OpenSSL context
    EVP_MD_CTX_free(ctx);
}
"""

# compile C code into shared library and load with ctypes
def _compile_hash_lib() -> str:
    import os, tempfile, subprocess

    # get OpenSSL include/lib paths from environment or use defaults
    openssl_prefix = os.environ.get('OPENSSL_PREFIX', '/usr')
    include_flag   = f'-I{openssl_prefix}/include'
    lib_flag       = f'-L{openssl_prefix}/lib64'   # lib64 on RHEL/Rocky/CentOS

    # write C source to a temporary file
    src = tempfile.NamedTemporaryFile(suffix='.c', delete=False, mode='w')
    src.write(_C_SOURCE)
    src.close()

    # compile to shared library
    lib_path = src.name.replace('.c', '.so')
    # flags for CPU on hmemq 
    subprocess.check_call([
        'gcc', '-O3', '-march=native', '-shared', '-fPIC',
        include_flag, lib_flag,
        '-o', lib_path, src.name,
        '-lcrypto',          # link OpenSSL libcrypto
    ])
    # cleanup source file
    os.unlink(src.name)
    # return path to compiled library
    return lib_path

_LIB_PATH = _compile_hash_lib()

# load library 
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

# hash-reduce a numpy array using the C library (when we need to apply the best reduction fucntion candidate)
def hash_reduce_numpy(col: np.ndarray, seed: int, N_val: int) -> np.ndarray:
    # aray to write to
    out = np.empty(len(col), dtype=np.uint64)
    # call c library from main process and hash and reduce the column
    _lib.hash_reduce_array(
        col.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_uint64(len(col)),
        ctypes.c_uint32(seed),
        ctypes.c_uint64(N_val),
        out.ctypes.data_as(ctypes.c_void_p),
    )
    #return the array
    return out


###############################################################################
# Worker functions

# per-worker global — holds the loaded C library handle
_worker_lib = None

# called once per wprker at pool creationand loads the c lbrary
def _worker_init(lib_path: str):
    # global vaiable to store loaded library handle so we don't reload every task 
    global _worker_lib
    # import again for worker
    import ctypes

    # load the library and set argtypes/restype for the hash_reduce_array function
    _worker_lib = ctypes.CDLL(lib_path)
    _worker_lib.hash_reduce_array.restype  = None
    _worker_lib.hash_reduce_array.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_uint32,
        ctypes.c_uint64,
        ctypes.c_void_p,
    ]


# hash and reduce a chunk of the column - for workers
def _hash_chunk(in_shm_name: str, out_shm_name: str, start: int, length: int, seed: int, N_val: int,):
    # imports for workers
    from multiprocessing import shared_memory
    import numpy as np
    import ctypes

    # access shared memory blocks by name
    in_shm  = shared_memory.SharedMemory(name=in_shm_name)
    out_shm = shared_memory.SharedMemory(name=out_shm_name)

    # wrap shared memory as numpy arrays so we can index into them — each worker gets a view into the relevant chunk
    in_arr  = np.ndarray((start + length,), dtype=np.uint64, buffer=in_shm.buf)
    out_arr = np.ndarray((start + length,), dtype=np.uint64, buffer=out_shm.buf)

    # get the chunk for this worker to process
    chunk_in  = in_arr[start : start + length]
    chunk_out = out_arr[start : start + length]

    # call the C library to hash-reduce the chunk
    _worker_lib.hash_reduce_array(
        chunk_in.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_uint64(length),
        ctypes.c_uint32(seed),
        ctypes.c_uint64(N_val),
        chunk_out.ctypes.data_as(ctypes.c_void_p),
    )

    # free shared memery
    in_shm.close()
    out_shm.close()


# to get the best seed for a column

def _count_seeds(col_shm_name: str, n_points: int, seed_batch: np.ndarray, N_val: int,):
    # if there are no seeds return -1
    if len(seed_batch) == 0:
        return None, -1
    
    # imports fro workers
    from multiprocessing import shared_memory
    import numpy as np
    import ctypes

    # access column from shared memory
    shm = shared_memory.SharedMemory(name=col_shm_name)
    col = np.ndarray((n_points,), dtype=np.uint64, buffer=shm.buf)

    # for best seeds
    best_count = -1
    best_seed  = int(seed_batch[0])

    # for each seed
    for s in seed_batch:
        # just in case
        s_int  = int(s)
        # empty array to write to
        output = np.empty(n_points, dtype=np.uint64)

        # worker reads from col_shm and write into output 
        _worker_lib.hash_reduce_array(
            col.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_uint64(n_points),
            ctypes.c_uint32(s_int),
            ctypes.c_uint64(N_val),
            output.ctypes.data_as(ctypes.c_void_p),
        )

        # np.unique on a pre-sorted array is O(n) — sort first
        # output.sort()
        # # count uniques via adjacent diff to avoid copying np array during unique
        # count = int(np.count_nonzero(np.diff(output))) + 1
        # del output

        # count unique
        count = np.unique(output).size
        del output

        if count > best_count:
            best_count = count
            best_seed  = s_int

    shm.close()
    return best_seed, best_count


###############################################################################
# parallel hash and reduce

def parallel_hash_reduce(pool: mp.Pool, col_shm: shared_memory.SharedMemory, out_shm: shared_memory.SharedMemory, n_points: int, seed: int, N_val: int, n_workers: int,) -> np.ndarray:

    # split column into chunks for each worker
    chunk_size = ceil(n_points / n_workers)
    args = []
    # for each worker
    for w in range(n_workers):
        # calculate start and length of chunk
        start  = w * chunk_size
        length = min(chunk_size, n_points - start)
        # if length is zero, we're done — no more chunks
        if length <= 0:
            break
        # args for workers
        args.append((col_shm.name, out_shm.name, start, length, seed, N_val))

    # run in prallerl 
    pool.starmap(_hash_chunk, args)

    # get from shared memory into a single numpy array to return
    result = np.ndarray((n_points,), dtype=np.uint64, buffer=out_shm.buf)
    return result.copy()   # copy so we own the memory independently of shm


###############################################################################
# parallel seed search 

# for each column, we have a batch of candidate seeds to evaluate.
def parallel_seed_search(pool:mp.Pool, col_shm:shared_memory.SharedMemory, n_points:int, index_sample:np.ndarray, N_val:int, n_workers:int,) -> int:
    # in case we have fewer candidates than workers
    n_workers = min(n_workers, len(index_sample))
    # split seeds into batches for each worker
    batches = np.array_split(index_sample, n_workers)
    # args for each worker
    args = [(col_shm.name, n_points, b, N_val) for b in batches]

    # run in parallel and get results
    results  = pool.starmap(_count_seeds, args)
    # filter out any None results (in case some workers got empty batches)
    results  = [(s, c) for s, c in results if s is not None]
    # get max by count
    best_seed, _ = max(results, key=lambda x: x[1])
    return best_seed


###############################################################################
# build rainbow table

def build(m0, t, kis):
    # start timer
    start_time = perf_counter()

    # to get unique seeds for each column
    t_bits      = (t - 1).bit_length()
    k_bits      = 32 - t_bits
    max_k_index = 2 ** k_bits

    # np arrays for current column, startpoints, and chosen rf indexes
    rf_indexes     = np.zeros(t, dtype=np.uint32)
    current_column = np.arange(round(m0), dtype=np.uint64)
    startpoints    = np.arange(round(m0), dtype=np.uint64)

    # put stuff in shared memory so we don't have to always copy
    # col_shm holds current_column (read by seed-search workers)
    # out_shm to write to during hashing and reducingh 
    # as the column shrinks we just use a prefix of each block to avoid reallocating shared memeory
    col_shm = shared_memory.SharedMemory(create=True, size=current_column.nbytes)
    out_shm = shared_memory.SharedMemory(create=True, size=current_column.nbytes)

    # wrap shared memory as numpy arrays so we can index into them
    col_shm_arr = np.ndarray(current_column.shape, dtype=np.uint64, buffer=col_shm.buf)

    # create a multiprocessing pool with the C library loaded in each worker's global scope.
    # workers load the C library in their initialiser and stay alive for all t iterations, so we only pay the loading cost once per worker.
    ctx = mp.get_context('spawn')

    # try block so we clean up shared memory even if something goes wrong in the pool
    try:
        # make sure pool closes properly
        with ctx.Pool(
            processes=CORES,
            initializer=_worker_init,
            initargs=(_LIB_PATH,),
        ) as pool:

            # for each column
            for i in range(t):
                # get the reduction function candidates
                k_i = round(kis[i])
                picks = np.array(random.sample(range(max_k_index), k_i), dtype=np.uint32)
                index_sample = (np.uint32(i) << np.uint32(k_bits)) | picks

                n_points = len(current_column)

                # find the best seed for this column by applying all candidates and counting uniques in parallel across cores.
                # write current column into col_shm so seed-search workers can read
                np.copyto(col_shm_arr[:n_points], current_column)

                # get the best seed index by splitting across cores
                best_index = parallel_seed_search(pool, col_shm, n_points, index_sample, N, CORES)
                rf_indexes[i] = best_index

                # apply the best reduction function 
                # workers read col_shm and write into out_shm concurrently.
                reduced = parallel_hash_reduce(
                    pool, col_shm, out_shm, n_points, int(best_index), N, CORES
                )

                # find unique endpoints and their corresponding startpoints
                unique_pts, sp_idx = np.unique(reduced, return_index=True)      # store unique points and their indexes to get their startpoints
                current_column = unique_pts                                     # update current column to unique endpoints
                startpoints = startpoints[sp_idx]                               # update startpoints to match unique endpoints

                # free memory
                del reduced, unique_pts, sp_idx

    finally:
        # cleanup shared memory
        col_shm.close(); col_shm.unlink()
        out_shm.close(); out_shm.unlink()

    # print duration and return results
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
