# Imports
from math import ceil
import cupy as cp
import numpy as np
import random
import multiprocessing as mp
from multiprocessing import shared_memory
from time import perf_counter


###############################################################################


# Constants
N     = 2 ** 40
alpha = 0.95
t     = 10000
gpus  = 4

def calculate_m0(N, t, alpha):
    mttarget = (2 * N) / (t + 2)
    return (alpha * mttarget) / (1 - alpha)

m0 = calculate_m0(N, t, alpha)

filename = "N40_kcurve.pkl"
kis = np.load(filename, allow_pickle=True)


###############################################################################
# CUDA kernel
#
# SHA-256 device code is taken verbatim from:
#   mochimodev/cuda-hashing-algos (sha256.cu)
#   Released into the Public Domain, 12 June 2019.
#   Based on Brad Conte's public domain C reference implementation.
#   https://github.com/mochimodev/cuda-hashing-algos
#
# MurmurHash3 device code is taken verbatim from:
#   aappleby/smhasher (MurmurHash3.cpp)
#   "MurmurHash3 was written by Austin Appleby, and is placed in the
#    public domain." — Austin Appleby
#   https://github.com/aappleby/smhasher
#
# Both implementations are unmodified except for:
#   - Removal of the host-side batch launcher (not needed here)
#   - Addition of the outer hash_reduce_kernel that wires them together
###############################################################################

KERNEL_CODE = r"""
// ============================================================
//  SHA-256  —  mochimodev/cuda-hashing-algos (public domain)
// ============================================================

typedef unsigned char  BYTE;
typedef unsigned int   WORD;

#define SHA256_BLOCK_SIZE 32

typedef struct {
    BYTE data[64];
    WORD datalen;
    unsigned long long bitlen;
    WORD state[8];
} CUDA_SHA256_CTX;

#ifndef ROTLEFT
#define ROTLEFT(a,b) (((a) << (b)) | ((a) >> (32-(b))))
#endif
#define ROTRIGHT(a,b) (((a) >> (b)) | ((a) << (32-(b))))

#define CH(x,y,z)  (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTRIGHT(x,2)  ^ ROTRIGHT(x,13) ^ ROTRIGHT(x,22))
#define EP1(x) (ROTRIGHT(x,6)  ^ ROTRIGHT(x,11) ^ ROTRIGHT(x,25))
#define SIG0(x) (ROTRIGHT(x,7) ^ ROTRIGHT(x,18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x,17)^ ROTRIGHT(x,19) ^ ((x) >> 10))

__constant__ WORD sha256_k[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,
    0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
    0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,
    0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,
    0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
    0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,
    0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,
    0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
    0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

__device__ __forceinline__
void cuda_sha256_transform(CUDA_SHA256_CTX *ctx, const BYTE data[])
{
    WORD a,b,c,d,e,f,g,h,i,j,t1,t2,m[64];
    for (i=0,j=0; i<16; ++i,j+=4)
        m[i] = (data[j]<<24)|(data[j+1]<<16)|(data[j+2]<<8)|(data[j+3]);
    for (; i<64; ++i)
        m[i] = SIG1(m[i-2]) + m[i-7] + SIG0(m[i-15]) + m[i-16];

    a=ctx->state[0]; b=ctx->state[1]; c=ctx->state[2]; d=ctx->state[3];
    e=ctx->state[4]; f=ctx->state[5]; g=ctx->state[6]; h=ctx->state[7];

    for (i=0; i<64; ++i) {
        t1 = h + EP1(e) + CH(e,f,g) + sha256_k[i] + m[i];
        t2 = EP0(a) + MAJ(a,b,c);
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    ctx->state[0]+=a; ctx->state[1]+=b; ctx->state[2]+=c; ctx->state[3]+=d;
    ctx->state[4]+=e; ctx->state[5]+=f; ctx->state[6]+=g; ctx->state[7]+=h;
}

__device__ void cuda_sha256_init(CUDA_SHA256_CTX *ctx)
{
    ctx->datalen = 0;
    ctx->bitlen  = 0;
    ctx->state[0] = 0x6a09e667; ctx->state[1] = 0xbb67ae85;
    ctx->state[2] = 0x3c6ef372; ctx->state[3] = 0xa54ff53a;
    ctx->state[4] = 0x510e527f; ctx->state[5] = 0x9b05688c;
    ctx->state[6] = 0x1f83d9ab; ctx->state[7] = 0x5be0cd19;
}

__device__ void cuda_sha256_update(CUDA_SHA256_CTX *ctx, const BYTE data[], size_t len)
{
    WORD i;
    for (i=0; i<len; ++i) {
        ctx->data[ctx->datalen] = data[i];
        ctx->datalen++;
        if (ctx->datalen == 64) {
            cuda_sha256_transform(ctx, ctx->data);
            ctx->bitlen += 512;
            ctx->datalen = 0;
        }
    }
}

__device__ void cuda_sha256_final(CUDA_SHA256_CTX *ctx, BYTE hash[])
{
    WORD i = ctx->datalen;

    if (ctx->datalen < 56) {
        ctx->data[i++] = 0x80;
        while (i < 56) ctx->data[i++] = 0x00;
    } else {
        ctx->data[i++] = 0x80;
        while (i < 64) ctx->data[i++] = 0x00;
        cuda_sha256_transform(ctx, ctx->data);
        for (int j=0; j<56; j++) ctx->data[j] = 0;
    }

    ctx->bitlen += ctx->datalen * 8;
    ctx->data[63] = ctx->bitlen;       ctx->data[62] = ctx->bitlen >> 8;
    ctx->data[61] = ctx->bitlen >> 16; ctx->data[60] = ctx->bitlen >> 24;
    ctx->data[59] = ctx->bitlen >> 32; ctx->data[58] = ctx->bitlen >> 40;
    ctx->data[57] = ctx->bitlen >> 48; ctx->data[56] = ctx->bitlen >> 56;
    cuda_sha256_transform(ctx, ctx->data);

    for (i=0; i<4; ++i) {
        hash[i]    = (ctx->state[0] >> (24 - i*8)) & 0xff;
        hash[i+4]  = (ctx->state[1] >> (24 - i*8)) & 0xff;
        hash[i+8]  = (ctx->state[2] >> (24 - i*8)) & 0xff;
        hash[i+12] = (ctx->state[3] >> (24 - i*8)) & 0xff;
        hash[i+16] = (ctx->state[4] >> (24 - i*8)) & 0xff;
        hash[i+20] = (ctx->state[5] >> (24 - i*8)) & 0xff;
        hash[i+24] = (ctx->state[6] >> (24 - i*8)) & 0xff;
        hash[i+28] = (ctx->state[7] >> (24 - i*8)) & 0xff;
    }
}


// ============================================================
//  MurmurHash3_x86_32  —  aappleby/smhasher (public domain)
// ============================================================

#define ROTL32(x,y) (((x) << (y)) | ((x) >> (32 - (y))))

__device__ __forceinline__ unsigned int mmh3_fmix32(unsigned int h)
{
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

// Hash `len` bytes starting at `key` with the given seed.
// Matches Python mmh3.hash(data, seed, signed=False).
__device__ unsigned int MurmurHash3_x86_32(const void* key, int len, unsigned int seed)
{
    const unsigned char* data    = (const unsigned char*)key;
    const int            nblocks = len / 4;

    unsigned int h1 = seed;
    const unsigned int c1 = 0xcc9e2d51u;
    const unsigned int c2 = 0x1b873593u;

    // body
    const unsigned int* blocks = (const unsigned int*)(data + nblocks*4);
    for (int i = -nblocks; i; i++) {
        unsigned int k1 = blocks[i];
        k1 *= c1;
        k1  = ROTL32(k1, 15);
        k1 *= c2;
        h1 ^= k1;
        h1  = ROTL32(h1, 13);
        h1  = h1*5 + 0xe6546b64u;
    }

    // tail
    const unsigned char* tail = data + nblocks*4;
    unsigned int k1 = 0;
    switch (len & 3) {
        case 3: k1 ^= (unsigned int)tail[2] << 16; // fall through
        case 2: k1 ^= (unsigned int)tail[1] << 8;  // fall through
        case 1: k1 ^= (unsigned int)tail[0];
                k1 *= c1; k1 = ROTL32(k1,15); k1 *= c2; h1 ^= k1;
    }

    // finalization
    h1 ^= len;
    return mmh3_fmix32(h1);
}


// ============================================================
//  Main kernel
// ============================================================
//
//  For each point x in input_data:
//    digest = SHA-256( x.to_bytes(8, 'little') )   [32 bytes]
//    result = MurmurHash3_x86_32(digest, 32, seed) % N_val
//
//  This exactly mirrors the original Python logic:
//    H_c(x)       = sha256(x.to_bytes(8,'little')).digest()
//    r_c(N,t,y,i) = mmh3.hash(y, seed, signed=False) % N

extern "C" __global__
void hash_reduce_kernel(
    const unsigned long long* __restrict__ input_data,
    long long                              n,
    unsigned int                           seed,
    unsigned long long                     N_val,
    unsigned long long* __restrict__       output)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Step 1: write input as 8 little-endian bytes
    unsigned long long val = input_data[idx];
    BYTE msg[8];
    msg[0] = (BYTE)( val        & 0xff);
    msg[1] = (BYTE)((val >>  8) & 0xff);
    msg[2] = (BYTE)((val >> 16) & 0xff);
    msg[3] = (BYTE)((val >> 24) & 0xff);
    msg[4] = (BYTE)((val >> 32) & 0xff);
    msg[5] = (BYTE)((val >> 40) & 0xff);
    msg[6] = (BYTE)((val >> 48) & 0xff);
    msg[7] = (BYTE)((val >> 56) & 0xff);

    // Step 2: SHA-256 -> 32-byte digest
    CUDA_SHA256_CTX ctx;
    BYTE digest[SHA256_BLOCK_SIZE];
    cuda_sha256_init(&ctx);
    cuda_sha256_update(&ctx, msg, 8);
    cuda_sha256_final(&ctx, digest);

    // Step 3: MurmurHash3 over the digest, reduce mod N
    unsigned int h = MurmurHash3_x86_32(digest, SHA256_BLOCK_SIZE, seed);
    output[idx] = (unsigned long long)h % N_val;
}
"""


###############################################################################
# main process kernel setup

# compiled once in the main process at startup.
_module = cp.RawModule(code=KERNEL_CODE, options=('-O3', '-arch=sm_80'), backend='nvcc')
_kernel_fn = _module.get_function('hash_reduce_kernel')

# take in an array and hash and reduce each item, returning the result as a new array
def hash_reduce_gpu(col_gpu: cp.ndarray, seed: int, N_val: int) -> cp.ndarray:
    """Run the hash-reduce kernel on an already-uploaded CuPy array."""
    n = col_gpu.size
    output = cp.empty(n, dtype=cp.uint64)      # empty array for the results
    threads = 256                              # threads per block 
    blocks = ceil(n / threads)                 # blocks per grid, rounded up to cover all n items
    # launch the kernel with (blocks, threads) configuration and the appropriate arguments
    _kernel_fn(
        (blocks,), (threads,),
        (col_gpu, np.int64(n), np.uint32(seed), np.uint64(N_val), output)
    )
    return output


###############################################################################

# per-worker globals for worker processes to store their assigned GPU device ID and compiled kernel function
_worker_device_id = None
_worker_fn = None

# initialiser function for worker processes — called once at pool startup, before any tasks are run
def _worker_init(id_queue, kernel_code: str):
    global _worker_device_id, _worker_fn
    # need to import cupy here inside the worker process after spawn, not at the top level of the module
    import cupy as cp

    # pop a unique GPU device ID from the queue
    _worker_device_id = id_queue.get()
    # set the CUDA device for this worker process
    with cp.cuda.Device(_worker_device_id):
        # compile the kernel once per worker process to avoid repeated nvcc compile cost
        mod = cp.RawModule(code=kernel_code, options=('-O3', '-arch=sm_80'), backend='nvcc')
        _worker_fn = mod.get_function('hash_reduce_kernel')




# def _worker_task(data_np: np.ndarray, seed_batch: np.ndarray, N_val: int):
#     # if there are no seeds to trial then return
#     if len(seed_batch) == 0:
#         return None, -1

#     # imports for worker
#     import cupy as cp
#     from math import ceil

#     with cp.cuda.Device(_worker_device_id):
#         # upload data to GPU
#         data_gpu = cp.asarray(data_np)
#         n        = data_gpu.size

#         # track best seed and number of unique points
#         best_count = -1
#         best_seed  = int(seed_batch[0])

#         # trial each seed
#         for s in seed_batch:
#             s_int   = int(s)    # just in case
#             output  = cp.empty(n, dtype=cp.uint64)  # array to store results
#             threads = 256   # number of threads
#             blocks  = ceil(n / threads) # number of blocks, rounded up to cover all n items
#             _worker_fn(
#                 (blocks,), (threads,),
#                 (data_gpu, np.int64(n), np.uint32(s_int), np.uint64(N_val), output)
#             )
#             # how many distinct points
#             count = int(cp.unique(output).size)
#             del output

#             # update
#             if count > best_count:
#                 best_count = count
#                 best_seed  = s_int

#         del data_gpu
#         cp.get_default_memory_pool().free_all_blocks()

#     return best_seed, best_count


# the worker task function — called once per column per worker, with a batch of seeds to evaluate
def _worker_task(shm_name: str, n_points: int, seed_batch: np.ndarray, N_val: int):

    if len(seed_batch) == 0:
        return None, -1
 
    import cupy as cp
    from math import ceil
    from multiprocessing import shared_memory
 
    with cp.cuda.Device(_worker_device_id):
 
        ## SHARE MEMORY INSTEAD OF TRANSFERRING DATA TO EACH WORKER
        # attach to shared memory 
        shm = shared_memory.SharedMemory(name=shm_name)
        data_np = np.ndarray((n_points,), dtype=np.uint64, buffer=shm.buf)
 
        # Upload to GPU once for all seeds in this batch
        data_gpu = cp.asarray(data_np)
        n = data_gpu.size
 
        # detach shm — GPU has its own copy now, shm no longer needed
        shm.close()
        ## FINISH SHARED MEMORY SETUP
 
        best_count = -1
        best_seed = int(seed_batch[0])
 
        for s in seed_batch:
            s_int = int(s)
            output = cp.empty(n, dtype=cp.uint64)
            threads = 256
            blocks = ceil(n / threads)
            _worker_fn(
                (blocks,), (threads,),
                (data_gpu, np.int64(n), np.uint32(s_int), np.uint64(N_val), output)
            )

            # count = int(cp.unique(output).size)
            # output.sort()
            # # count unique by comparing adjacent elements
            # diff = output[1:] != output[:-1]
            # count = int(diff.sum()) + 1
            # del diff
            # del output

            output_cpu = output.get()
            del output
            cp.get_default_memory_pool().free_all_blocks()

            count = len(np.unique(output_cpu))
            del output_cpu
 
            if count > best_count:
                best_count = count
                best_seed = s_int
 
        del data_gpu
        cp.get_default_memory_pool().free_all_blocks()
 
    return best_seed, best_count


###############################################################################

# build rainbow table
def build(m0, t, kis):
    # start timer
    start_time = perf_counter()

    # set up for getting indexes
    t_bits      = (t - 1).bit_length()
    k_bits      = 32 - t_bits
    max_k_index = 2 ** k_bits

    # array to hold the chosen reduction function index for each column
    rf_indexes     = np.zeros(t, dtype=np.uint32)
    current_column = np.arange(round(m0), dtype=np.uint64)
    startpoints    = np.arange(round(m0), dtype=np.uint64)

    # allocate shared memory sized for the initial (largest) column.
    # as the column shrinks each iteration we reuse the same block
    # we just tell workers the current live size via n_points.
    shm = shared_memory.SharedMemory(create=True, size=current_column.nbytes)
    shm_array = np.ndarray(current_column.shape, dtype=np.uint64, buffer=shm.buf)

    # Pre-load a queue so each worker claims a unique GPU device ID
    manager  = mp.Manager()
    id_queue = manager.Queue()
    for d in range(gpus):
        id_queue.put(d)

    ctx = mp.get_context('spawn')

    # Pool created ONCE — workers compile the kernel at startup and stay alive
    # for all t columns
    # try finally block to ensure shared memory is cleaned up even if something goes wrong in the pool
    try:
        # pool created once, workers stay alive and reuse their GPU contexts and compiled kernels for all columns
        with ctx.Pool(
            processes=gpus,
            initializer=_worker_init,
            initargs=(id_queue, KERNEL_CODE),
        ) as pool:

            # for all columns
            for i in range(t):
                # get the reduction functions to trial
                k_i          = round(kis[i])
                picks        = np.array(random.sample(range(max_k_index), k_i), dtype=np.uint32)
                index_sample = (np.uint32(i) << np.uint32(k_bits)) | picks

                # write the current column into shared memory for workers to access
                n_points = len(current_column)
                np.copyto(shm_array[:n_points], current_column)

                # Never dispatch more workers than there are seeds
                n_workers = min(gpus, len(index_sample))
                batches = np.array_split(index_sample, n_workers)
                # args      = [(current_column, batches[j], N) for j in range(n_workers)]
                # instead of passing the column data to each worker, we just pass the shared memory name and size
                args = [ (shm.name, n_points, batches[j], N) for j in range(n_workers)]

                # get the results from all the workers
                results = pool.starmap(_worker_task, args)
                results = [(s, c) for s, c in results if s is not None]

                # find best seed
                best_index = max(results, key=lambda x: x[1])[0]
                rf_indexes[i] = best_index

                # apply the winning reduction function on GPU 0
                with cp.cuda.Device(0):
                    col_gpu  = cp.asarray(current_column)
                    reduced  = hash_reduce_gpu(col_gpu, int(best_index), N)
                    del col_gpu

                    # get unique endpoints and corresponding startpoints for the next iteration
                    unique_pts, sp_idx = cp.unique(reduced, return_index=True)
                    current_column = unique_pts.get()
                    startpoints = startpoints[sp_idx.get()]

                    del reduced, unique_pts, sp_idx
                    cp.get_default_memory_pool().free_all_blocks()
    finally:
        # clean up shared memory
        shm.close()
        shm.unlink()


    duration = perf_counter() - start_time
    print(f"\nRainbow table built in {duration:.2f} seconds")

    return startpoints, current_column, rf_indexes


###############################################################################

if __name__ == '__main__':
    startpoints, final_column, rf_indexes = build(m0, t, kis)

    # store startpoints, endpoints, and rf indexes for later analysis
    np.savez(
        "216_t80_results.npz",
        startpoints=startpoints,
        endpoints=final_column,
        rf_indexes=rf_indexes,
    )

    print(f"\nFinal column length          : {len(final_column):,}")
    print(f"Startpoints / endpoints match: {len(startpoints) == len(final_column)}")
    print(f"Endpoints are unique         : {len(final_column) == len(set(final_column))}")
    print(f"Distinct rf_indexes          : {len(set(rf_indexes))} / {len(rf_indexes)}")