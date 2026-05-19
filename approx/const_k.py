import ctypes
from math import ceil
from multiprocessing import shared_memory
import multiprocessing as mp
import os
from time import perf_counter
import random
import numpy as np
# import pandas as pd
import csv


###############################################################################
# Constants
###############################################################################

N     = 20000
alpha = 0.8
t     = 100
CORES = 16          # physical cores on the high-mem node

# Memory budget for seed-search workers.
# Each worker allocates one uint64 output array (~16 GB at full m0).
# Fixed shared memory: col_shm (16 GB) + digest_shm (64 GB) = 80 GB.
# Remaining for workers: 1400 GB - 80 GB = 1320 GB.
# Max concurrent workers: 1320 GB / 16 GB = 82. Use 80 for safety.
SEED_WORKERS = 80

def calculate_m0(N, t, alpha):
    mttarget = (2 * N) / (t + 2)
    return (alpha * mttarget) / (1 - alpha)

m0 = calculate_m0(N, t, alpha)

# constant k_i - t lots of 100
kis = np.load("cherry_k_values.npy")

# what do we need to store?
# num hashes, num reductions, time taken to hash, time taken to reduce, time taken to de-duplicate
num_hashes = 0
num_reductions = 0
hash_time = 0
reduce_time = 0
dedup_time = 0


###############################################################################
# C library
#
# Three exported functions:
#
#   sha256_array(input, n, output)
#     SHA-256 each uint64 point (as 8 little-endian bytes) and write the
#     32-byte digest into output. Called once per column, parallelised
#     across CORES workers by chunk.
#     output must be uint8[n * 32].
#
#   reduce_array(digests, n, seed, N_val, output)
#     MurmurHash3_x86_32(digest[i], 32, seed) % N_val for each point.
#     Called once per seed candidate during seed search — no SHA-256.
#     digests is uint8[n * 32], output is uint64[n].
#
#   hash_reduce_array(input, n, seed, N_val, output)
#     SHA-256 + MurmurHash3 + reduce in one pass. Used in the final
#     apply step (best seed applied to get next column).
#     input is uint64[n], output is uint64[n].
#
# SHA-256: OpenSSL EVP API — uses SHA-NI hardware automatically.
# MurmurHash3: aappleby/smhasher reference, public domain.
###############################################################################

_C_SOURCE = r"""
#include <stdint.h>
#include <string.h>
#include <openssl/evp.h>

// ── MurmurHash3_x86_32  —  aappleby/smhasher (public domain) ────────────────

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


// ── Helper: write uint64 as 8 little-endian bytes ───────────────────────────

static inline void u64_to_le8(uint64_t val, uint8_t out[8])
{
    out[0] = (uint8_t)( val        & 0xff);
    out[1] = (uint8_t)((val >>  8) & 0xff);
    out[2] = (uint8_t)((val >> 16) & 0xff);
    out[3] = (uint8_t)((val >> 24) & 0xff);
    out[4] = (uint8_t)((val >> 32) & 0xff);
    out[5] = (uint8_t)((val >> 40) & 0xff);
    out[6] = (uint8_t)((val >> 48) & 0xff);
    out[7] = (uint8_t)((val >> 56) & 0xff);
}


// ── sha256_array: hash each uint64 point, write 32-byte digests ──────────────
//
// input   : uint64[n]       — the current column
// n       : number of points
// output  : uint8[n * 32]   — packed digest array; digest i is at output[i*32]
//
// Called once per column. Workers each process a contiguous chunk.

void sha256_array(
    const uint64_t *input,
    uint64_t        n,
    uint8_t        *output)
{
    EVP_MD_CTX   *ctx = EVP_MD_CTX_new();
    const EVP_MD *md  = EVP_sha256();

    for (uint64_t i = 0; i < n; ++i) {
        uint8_t msg[8];
        u64_to_le8(input[i], msg);

        unsigned int dlen = 32;
        EVP_DigestInit_ex(ctx, md, NULL);
        EVP_DigestUpdate(ctx, msg, 8);
        EVP_DigestFinal_ex(ctx, output + i * 32, &dlen);
    }

    EVP_MD_CTX_free(ctx);
}


// ── reduce_array: MurmurHash3 + mod over pre-computed digests ────────────────
//
// digests : uint8[n * 32]   — packed digest array from sha256_array
// n       : number of points
// seed    : MurmurHash3 seed (encodes column index + rf sub-index)
// N_val   : universe size; output values in [0, N_val)
// output  : uint64[n]       — reduced points
//
// Called once per seed candidate. No SHA-256 — cheap MurmurHash3 only.

void reduce_array(
    const uint8_t  *digests,
    uint64_t        n,
    uint32_t        seed,
    uint64_t        N_val,
    uint64_t       *output)
{
    for (uint64_t i = 0; i < n; ++i) {
        uint32_t h = MurmurHash3_x86_32(digests + i * 32, 32, seed);
        output[i]  = (uint64_t)h % N_val;
    }
}


// ── hash_reduce_array: SHA-256 + MurmurHash3 + reduce in one pass ────────────
//
// Used for the final apply step (applying the best seed to advance the column).
// Avoids storing digests since we only need the reduced output.
//
// input  : uint64[n]
// output : uint64[n]

void hash_reduce_array(
    const uint64_t *input,
    uint64_t        n,
    uint32_t        seed,
    uint64_t        N_val,
    uint64_t       *output)
{
    EVP_MD_CTX   *ctx = EVP_MD_CTX_new();
    const EVP_MD *md  = EVP_sha256();

    for (uint64_t i = 0; i < n; ++i) {
        uint8_t msg[8];
        u64_to_le8(input[i], msg);

        uint8_t      digest[32];
        unsigned int dlen = 32;
        EVP_DigestInit_ex(ctx, md, NULL);
        EVP_DigestUpdate(ctx, msg, 8);
        EVP_DigestFinal_ex(ctx, digest, &dlen);

        uint32_t h = MurmurHash3_x86_32(digest, 32, seed);
        output[i]  = (uint64_t)h % N_val;
    }

    EVP_MD_CTX_free(ctx);
}

// Add to C source — count uniques using a 32-bit bitmap
// Since MurmurHash3_x86_32 produces 32-bit output, values before % N_val
// are in [0, 2^32). A bitmap of 2^32 bits = 512 MB lets us count uniques
// in a single O(n) pass with no sorting.

void reduce_count_unique(
    const uint8_t  *digests,
    uint64_t        n,
    uint32_t        seed,
    uint64_t        N_val,
    uint64_t       *out_count)   // single uint64 written with the unique count
{
    // 2^32 bits = 512 MB bitmap, zero-initialised
    static const uint64_t BITMAP_WORDS = (1ULL << 32) / 64;
    uint64_t *bitmap = (uint64_t *)calloc(BITMAP_WORDS, sizeof(uint64_t));
    if (!bitmap) { *out_count = 0; return; }

    uint64_t count = 0;

    for (uint64_t i = 0; i < n; ++i) {
        // MurmurHash3 gives a 32-bit value — use it directly before % N_val
        uint32_t h    = MurmurHash3_x86_32(digests + i * 32, 32, seed);
        uint64_t word = h >> 6;          // which 64-bit word
        uint64_t bit  = 1ULL << (h & 63); // which bit within the word
        if (!(bitmap[word] & bit)) {
            bitmap[word] |= bit;
            count++;
        }
    }

    free(bitmap);
    *out_count = count;
}


"""


###############################################################################
# Compile and load C library
###############################################################################

def _compile_hash_lib() -> str:
    import tempfile, subprocess

    openssl_prefix = os.environ.get('OPENSSL_PREFIX', '/usr')
    include_flag   = f'-I{openssl_prefix}/include'
    lib_flag       = f'-L{openssl_prefix}/lib'

    src = tempfile.NamedTemporaryFile(suffix='.c', delete=False, mode='w')
    src.write(_C_SOURCE)
    src.close()

    lib_path = src.name.replace('.c', '.so')
    subprocess.check_call([
        'gcc', '-O3', '-march=native', '-shared', '-fPIC',
        include_flag, lib_flag,
        '-o', lib_path, src.name,
        '-lcrypto',
    ])
    os.unlink(src.name)
    return lib_path


def _load_lib(path: str) -> ctypes.CDLL:
    lib = ctypes.CDLL(path)

    # sha256_array(input, n, output)
    lib.sha256_array.restype  = None
    lib.sha256_array.argtypes = [
        ctypes.c_void_p,   # input  uint64*
        ctypes.c_uint64,   # n
        ctypes.c_void_p,   # output uint8*  (n * 32 bytes)
    ]

    # reduce_array(digests, n, seed, N_val, output)
    lib.reduce_array.restype  = None
    lib.reduce_array.argtypes = [
        ctypes.c_void_p,   # digests uint8*
        ctypes.c_uint64,   # n
        ctypes.c_uint32,   # seed
        ctypes.c_uint64,   # N_val
        ctypes.c_void_p,   # output  uint64*
    ]

    # hash_reduce_array(input, n, seed, N_val, output)
    lib.hash_reduce_array.restype  = None
    lib.hash_reduce_array.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_uint32,
        ctypes.c_uint64,
        ctypes.c_void_p,
    ]

    return lib


_LIB_PATH = _compile_hash_lib()
_lib       = _load_lib(_LIB_PATH)


###############################################################################
# Main-process helpers
###############################################################################

# def hash_reduce_numpy(col: np.ndarray, seed: int, N_val: int) -> np.ndarray:
#     """SHA-256 + MurmurHash3 + reduce for the final apply step."""

#     out = np.empty(len(col), dtype=np.uint64)
#     _lib.hash_reduce_array(
#         col.ctypes.data_as(ctypes.c_void_p),
#         ctypes.c_uint64(len(col)),
#         ctypes.c_uint32(seed),
#         ctypes.c_uint64(N_val),
#         out.ctypes.data_as(ctypes.c_void_p),
#     )
#     return out

from numba import njit

@njit
def radix_sort(arr):
    # Find the maximum number to know number of digits
    max_val = arr.max()
    exp = 1
    n = len(arr)
    output = np.empty_like(arr)

    while max_val // exp > 0:
        # Counting sort for the current digit (exp)
        count = np.zeros(10, dtype=np.intp)
        
        for i in range(n):
            index = (arr[i] // exp) % 10
            count[index] += 1
            
        for i in range(1, 10):
            count[i] += count[i - 1]
            
        for i in range(n - 1, -1, -1):
            index = (arr[i] // exp) % 10
            output[count[index] - 1] = arr[i]
            count[index] -= 1
            
        for i in range(n):
            arr[i] = output[i]
            
        exp *= 10
    return arr

###############################################################################
# Worker globals and initialiser
###############################################################################

_worker_lib = None

def _worker_init(lib_path: str):
    """Load the C library once per worker process at pool startup."""
    global _worker_lib
    import ctypes
    _worker_lib = ctypes.CDLL(lib_path)

    _worker_lib.sha256_array.restype  = None
    _worker_lib.sha256_array.argtypes = [
        ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p,
    ]
    _worker_lib.reduce_array.restype  = None
    _worker_lib.reduce_array.argtypes = [
        ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32,
        ctypes.c_uint64, ctypes.c_void_p,
    ]
    _worker_lib.hash_reduce_array.restype  = None
    _worker_lib.hash_reduce_array.argtypes = [
        ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32,
        ctypes.c_uint64, ctypes.c_void_p,
    ]


###############################################################################
# Phase 1 worker: SHA-256 a chunk of the column into digest_shm
###############################################################################

def _sha256_chunk(
    col_shm_name:    str,
    digest_shm_name: str,
    start:           int,
    length:          int,
):
    """
    SHA-256 input[start : start+length] and write 32-byte digests into
    digest_shm[start*32 : (start+length)*32].
    No seed needed — digests are seed-independent.
    """
    from multiprocessing import shared_memory
    import numpy as np
    import ctypes

    col_shm    = shared_memory.SharedMemory(name=col_shm_name)
    digest_shm = shared_memory.SharedMemory(name=digest_shm_name)

    col_arr    = np.ndarray((start + length,), dtype=np.uint64, buffer=col_shm.buf)
    digest_arr = np.ndarray(((start + length) * 32,), dtype=np.uint8, buffer=digest_shm.buf)

    chunk_in  = col_arr[start : start + length]
    chunk_out = digest_arr[start * 32 : (start + length) * 32]

    # start time
    start = perf_counter()

    _worker_lib.sha256_array(
        chunk_in.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_uint64(length),
        chunk_out.ctypes.data_as(ctypes.c_void_p),
    )

    # add time taken to hash
    global hash_time
    hash_time += perf_counter() - start

    # add number of hashes processed
    global num_hashes
    num_hashes += length

    col_shm.close()
    digest_shm.close()


###############################################################################
# Phase 2 worker: MurmurHash3 + reduce over digests for a batch of seeds
###############################################################################

def _count_seeds(
    digest_shm_name: str,
    n_points:        int,
    seed_batch:      np.ndarray,
    N_val:           int,
):
    """
    For each seed in seed_batch:
      - Read pre-computed digests from digest_shm (zero copy, no SHA-256)
      - Apply MurmurHash3 + reduce mod N_val
      - Count unique output values
    Returns (best_seed, best_count).

    Peak memory per worker: one uint64 output array = n_points * 8 bytes.
    """
    if len(seed_batch) == 0:
        return None, -1

    from multiprocessing import shared_memory
    import numpy as np
    import ctypes

    digest_shm = shared_memory.SharedMemory(name=digest_shm_name)
    digests    = np.ndarray((n_points * 32,), dtype=np.uint8, buffer=digest_shm.buf)

    best_count = -1
    best_seed  = int(seed_batch[0])

    for s in seed_batch:
        s_int  = int(s)
        output = np.empty(n_points, dtype=np.uint64)

        # start time to reduce
        start = perf_counter()

        _worker_lib.reduce_array(
            digests.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_uint64(n_points),
            ctypes.c_uint32(s_int),
            ctypes.c_uint64(N_val),
            output.ctypes.data_as(ctypes.c_void_p),
        )

        # add time taken to reduce
        global reduce_time
        reduce_time += perf_counter() - start

        # add number of reductions processed        
        global num_reductions
        num_reductions += n_points

        # start time to de-duplicate
        start = perf_counter()

        output = radix_sort(output)
        count = int(np.count_nonzero(np.diff(output))) + 1

        # add time taken to de-duplicate
        global dedup_time
        dedup_time += perf_counter() - start

        del output

        if count > best_count:
            best_count = count
            best_seed  = s_int

    digest_shm.close()
    return best_seed, best_count


###############################################################################
# Parallel SHA-256 phase
###############################################################################

def parallel_sha256(
    pool:            mp.Pool,
    col_shm:         shared_memory.SharedMemory,
    digest_shm:      shared_memory.SharedMemory,
    n_points:        int,
    n_workers:       int,
):
    """
    Hash the current column in parallel. Each worker SHA-256s its chunk
    and writes into the corresponding slice of digest_shm.
    On return, digest_shm contains SHA-256(x) for every x in current_column.
    """
    chunk_size = ceil(n_points / n_workers)
    args = []
    for w in range(n_workers):
        start  = w * chunk_size
        length = min(chunk_size, n_points - start)
        if length <= 0:
            break
        args.append((col_shm.name, digest_shm.name, start, length))

    pool.starmap(_sha256_chunk, args)


###############################################################################
# Parallel seed search phase
###############################################################################

def parallel_seed_search(
    pool:            mp.Pool,
    digest_shm:      shared_memory.SharedMemory,
    n_points:        int,
    index_sample:    np.ndarray,
    N_val:           int,
    n_workers:       int,
) -> int:
    """
    Split seed candidates across n_workers cores.
    Each worker reads digests from digest_shm (zero copy) and evaluates
    its batch of seeds using only MurmurHash3 + reduce — no SHA-256.
    Returns the seed with the most unique reduced points.
    """
    n_workers = min(n_workers, len(index_sample))
    batches   = np.array_split(index_sample, n_workers)
    args      = [(digest_shm.name, n_points, b, N_val) for b in batches]

    results  = pool.starmap(_count_seeds, args)
    results  = [(s, c) for s, c in results if s is not None]
    best_seed, _ = max(results, key=lambda x: x[1])
    return best_seed


###############################################################################
# Rainbow table builder
###############################################################################

def build(m0, t, kis):

    # store m_values
    m_values = np.empty(t + 1, dtype=np.uint64)
    m_values[0] = m0

    start_time = perf_counter()

    t_bits      = (t - 1).bit_length()
    k_bits      = 32 - t_bits
    max_k_index = 2 ** k_bits

    rf_indexes     = np.zeros(t, dtype=np.uint32)
    current_column = np.arange(round(m0), dtype=np.uint64)
    startpoints    = np.arange(round(m0), dtype=np.uint64)

    n_initial = len(current_column)

    # ── Shared memory ─────────────────────────────────────────────────────────
    # col_shm    : current_column as uint64 — ~16 GB at m0
    # digest_shm : SHA-256 digests as uint8 — ~64 GB at m0 (32 bytes per point)
    # Both sized for the initial column. As the column shrinks each iteration
    # we reuse the same blocks, passing n_points to track the live prefix.
    col_shm    = shared_memory.SharedMemory(create=True, size=current_column.nbytes)
    digest_shm = shared_memory.SharedMemory(create=True, size=n_initial * 32)

    col_shm_arr = np.ndarray(current_column.shape, dtype=np.uint64, buffer=col_shm.buf)

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

                # ── Phase 1: SHA-256 the column once, in parallel ─────────────
                # Write current column into col_shm for workers to read.
                np.copyto(col_shm_arr[:n_points], current_column)

                # All CORES workers hash their chunk into digest_shm.
                # After this, digest_shm[i*32 : i*32+32] = SHA256(current_column[i])
                parallel_sha256(pool, col_shm, digest_shm, n_points, CORES)

                # ── Phase 2: find best seed using digests only ────────────────
                # Workers read digest_shm and apply MurmurHash3 + reduce.
                # No SHA-256 here — orders of magnitude cheaper per seed.
                # SEED_WORKERS capped to stay within memory budget.
                best_index    = parallel_seed_search(
                    pool, digest_shm, n_points, index_sample, N, SEED_WORKERS
                )
                rf_indexes[i] = best_index

                # ── Phase 3: apply best reduction function ────────────────────
                # Use col_shm (already written) to compute the next column.
                # hash_reduce_array does SHA-256 + MurmurHash3 in one pass —
                # no need to store digests for this step.
                # reduced = hash_reduce_numpy(current_column, int(best_index), N)

                # reuse digests to apply best reduction function without storing digests separately
                # digest_shm is already populated by parallel_sha256
                digests = np.ndarray((n_points * 32,), dtype=np.uint8, buffer=digest_shm.buf)
                reduced = np.empty(n_points, dtype=np.uint64)

                # add time taken to reduce
                start = perf_counter()

                _lib.reduce_array(
                    digests.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_uint64(n_points),
                    ctypes.c_uint32(int(best_index)),
                    ctypes.c_uint64(N),
                    reduced.ctypes.data_as(ctypes.c_void_p),
                )

                global reduce_time, num_reductions
                reduce_time += perf_counter() - start

                # add number of reductions processed
                num_reductions += n_points


                # add time taken to de-duplicate
                start = perf_counter()

                unique_pts, sp_idx = np.unique(reduced, return_index=True)

                global dedup_time
                dedup_time += perf_counter() - start

                m_values[i+1] = len(unique_pts)

                current_column     = unique_pts
                startpoints        = startpoints[sp_idx]

                del reduced, unique_pts, sp_idx

                if i % 10 == 0:
                    elapsed = perf_counter() - start_time
                    print(
                        f"  col {i:5d}/{t}  survivors={len(current_column):,}  "
                        f"elapsed={elapsed:.1f}s"
                    )

    finally:
        col_shm.close();    col_shm.unlink()
        digest_shm.close(); digest_shm.unlink()

    duration = perf_counter() - start_time
    print(f"\nRainbow table built in {duration:.2f} seconds")
    return startpoints, current_column, rf_indexes, m_values


###############################################################################



if __name__ == '__main__':
    # startpoints, final_column, rf_indexes, m_values = build(m0, t, kis)

    # output_dir = "results"
    # os.makedirs(output_dir, exist_ok=True)

    # np.save(f"{output_dir}/startpoints.npy",  startpoints)
    # np.save(f"{output_dir}/final_column.npy", final_column)
    # np.save(f"{output_dir}/rf_indexes.npy",   rf_indexes)
    # np.save(f"{output_dir}/m_values.npy",     m_values)


    # # store in a file the metadata about the run: num_hashes, num_reductions, hash_time, reduce_time, dedup_time
    # import pandas as pd
    # metadata = {
    #     "endpoints": len(final_column),
    #     "num_hashes": num_hashes,
    #     "num_reductions": num_reductions,
    #     "hash_time": hash_time,
    #     "reduce_time": reduce_time,
    #     "dedup_time": dedup_time,
    #     "hash_rate": num_hashes / hash_time if hash_time > 0 else 0,
    #     "reduce_rate": num_reductions / reduce_time if reduce_time > 0 else 0,
    # }
    # df = pd.DataFrame([metadata])
    # df.to_csv(f"{output_dir}/metadata.csv", index=False)

    # # store m_values
    # np.save(f"{output_dir}/m_values.npy", m_values)



    # print(f"\nResults saved to {output_dir}/")
    # print(f"Final column length          : {len(final_column):,}")
    # print(f"Startpoints / endpoints match: {len(startpoints) == len(final_column)}")
    # print(f"Endpoints are unique         : {len(final_column) == len(set(final_column))}")
    # print(f"Distinct rf_indexes          : {len(set(rf_indexes))} / {len(rf_indexes)}")

    # have a csv instead of pandas dataframe to store the results of multiple runs with different random seeds
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    # make csv file with the columns
    csv_file = f"{output_dir}/results.csv"
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['trial', 'endpoints', 'num_hashes', 'num_reductions', 'hash_time', 'reduce_time', 'dedup_time', 'hash_rate', 'reduce_rate'])


    # results_df = pd.DataFrame()
    # # columns: trial, endpoints, num_hashes, num_reductions, hash_time, reduce_time, dedup_time, hash_rate, reduce_rate
    # results_df.columns = ['trial', 'endpoints', 'num_hashes', 'num_reductions', 'hash_time', 'reduce_time', 'dedup_time', 'hash_rate', 'reduce_rate']

    # build 10 samples of tables
    for i in range(10):

        # reset global counters
        num_hashes = 0
        num_reductions = 0
        hash_time = 0
        reduce_time = 0
        dedup_time = 0

        # build the table
        startpoints, final_column, rf_indexes, m_values = build(m0, t, kis)

        # # store details about the run in the dataframe
        # results_df = results_df.append({
        #     'trial': i,
        #     'endpoints': len(final_column),
        #     'num_hashes': num_hashes,
        #     'num_reductions': num_reductions,
        #     'hash_time': hash_time,
        #     'reduce_time': reduce_time,
        #     'dedup_time': dedup_time,
        #     'hash_rate': num_hashes / hash_time if hash_time > 0 else 0,
        #     'reduce_rate': num_reductions / reduce_time if reduce_time > 0 else 0,
        # }, ignore_index=True)

        # write details about the run to the csv file
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                i,
                len(final_column),
                num_hashes,
                num_reductions,
                hash_time,
                reduce_time,
                dedup_time,
                num_hashes / hash_time if hash_time > 0 else 0,
                num_reductions / reduce_time if reduce_time > 0 else 0,
            ])

        # save startpoints, final_column, rf_indexes, m_values for this trial
        trial_dir = f"{output_dir}/trial_{i}"
        os.makedirs(trial_dir, exist_ok=True)
        np.save(f"{trial_dir}/startpoints.npy",  startpoints)
        np.save(f"{trial_dir}/final_column.npy", final_column)
        np.save(f"{trial_dir}/rf_indexes.npy",   rf_indexes)
        np.save(f"{trial_dir}/m_values.npy",     m_values)


