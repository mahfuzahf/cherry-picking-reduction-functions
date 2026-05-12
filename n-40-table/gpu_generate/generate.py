# Imports
from math import ceil
import cupy as cp
import numpy as np
from numba import cuda
import mmh3
from hashlib import sha256
import random
import multiprocessing as mp
from time import perf_counter


###############################################################################


# Constants
N = 2 ** 16
alpha = 0.95
t = 80
gpus = 2

# calculate m_0 
def calculate_m0(N, t, alpha):
    mttarget = (2 * N) / (t + 2)
    return (alpha * mttarget) / (1 - alpha)

m0 = calculate_m0(N, t, alpha)


# import pre-optimised k_i values
filename = "data/N_16_alpha_0.95_cost_5_t_80_optimised_kis.pkl"
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

# Compile once in the main process.
# backend='nvcc' is required for structs, __constant__, and #define to work.
_module = cp.RawModule(
    code=KERNEL_CODE,
    options=('-O3', '-arch=sm_80'),  # A100 = sm_80
    backend='nvcc',
)
_kernel_fn = _module.get_function('hash_reduce_kernel')
 
 
def hash_reduce_gpu(col_gpu: cp.ndarray, seed: int, N_val: int) -> cp.ndarray:
    """Run the hash-reduce kernel on an already-uploaded CuPy array."""
    n       = col_gpu.size
    output  = cp.empty(n, dtype=cp.uint64)
    threads = 256
    blocks  = ceil(n / threads)
    _kernel_fn(
        (blocks,), (threads,),
        (col_gpu, np.int64(n), np.uint32(seed), np.uint64(N_val), output)
    )
    return output

###############################################################################

# # Hash and reduction functions
# @cuda.jit(target_backend='cuda')
# def H_c(x):
#     return sha256(x.to_bytes(8, 'little')).digest()  # return bytes directly

# @cuda.jit(target_backend='cuda')
# def r_c(N, t, y, i, ell=0):
#     seed = int(i) + (ell*t)
#     return mmh3.hash(y, seed, signed=False) % N

# @cuda.jit(target_backend='cuda')
# def hash_reduce_kernel(input_data, seed, output_points):
#     idx = cuda.grid(1)
#     if idx < input_data.size:

#         # hash and reduce to get the next point
#         point = r_c(N, t, H_c(input_data[idx]), seed)
        
#         output_points[idx] = point


###############################################################################


def worker_task(device_id: int, data_np: np.ndarray, seed_batch: np.ndarray):
    """Function run by each of the 8 processes."""

    import cupy as cp          # re-import in the spawned process
    from math import ceil

    # assign this process to a specific A100
    with cp.cuda.Device(device_id):

        # Compile once per worker process
        mod = cp.RawModule(
            code=KERNEL_CODE,
            options=('-O3', '-arch=sm_80'),
            backend='nvcc',
        )
        fn  = mod.get_function('hash_reduce_kernel')

        # upload once per process to avoid repeated transferss
        data_gpu = cp.asarray(data_np)
        n = data_gpu.size
        
        local_best_len = -1
        local_best_seed = int(seed_batch[0])
        
        for s in seed_batch:

            # convert seed to int
            s_int = int(s)


            # Allocate temp hash space (32GB)
            # points = cp.empty(data_gpu.size, dtype=cp.uint64)
            
            # # Run Kernel
            # threads_per_block = 256                                                         # how many individual threads grouped together in a single block
            # blocks_per_grid = ceil(data_gpu.size / threads_per_block)                       # how many blocks we need to cover all data
            # hash_reduce_kernel[blocks_per_grid, threads_per_block](data_gpu, s, points)     # launch the kernel with the specified grid and block dimensions

            # allocate temp space
            output  = cp.empty(n, dtype=cp.uint64)
            threads = 256
            blocks  = ceil(n / threads)
            fn(
                (blocks,), (threads,),
                (data_gpu, np.int64(n), np.uint32(s), np.uint64(N), output)
            )

            # get exact count of unique points in output (after hash-reduce)
            count = cp.unique(output).size
            del output
 
            if count > local_best_len:
                local_best_len = count
                local_best_seed  = s
            
            # # exact count 
            # count = cp.unique(points).size
            
            # if count > local_best_len:
            #     local_best_len = count
            #     local_best_seed = s
            
            # Cleanup to prevent OOM
            # del points
        
        # Cleanup data before returning
        del data_gpu
        cp.get_default_memory_pool().free_all_blocks()
        
        return local_best_seed, local_best_len


def get_best_seed_distributed(data_np: np.ndarray, all_seeds: np.ndarray):
    # split seeds into 8 batches for 8 GPUs
    seed_batches = np.array_split(all_seeds, gpus)
    
    # launch the worker function on each GPU in parallel
    # 'spawn' to ensure clean CUDA contexts
    ctx = mp.get_context('spawn')
    with ctx.Pool(gpus) as pool:
        # Prepare arguments for each GPU (device_id, data, batch)
        args = [(i, data_np, seed_batches[i]) for i in range(gpus)]
        
        # 3. Parallel Execution
        results = pool.starmap(worker_task, args)
    
    # 4. Find the global winner from the 8 local winners
    global_best_seed, _ = max(results, key=lambda x: x[1])
    return global_best_seed


###############################################################################


# build the rainbow table
def build(m0, t, kis):

    # start timer
    start_time = perf_counter()

    # get parameters for reduction function search
    t_bits = (t-1).bit_length()                     # number of bits to represent t
    k_bits = 32 - t_bits                            # number of bits to represent rf index
    max_k_index = 2**k_bits                         # maximum number of reduction functions per column

    # np array to store fr_indexes
    rf_indexes = np.zeros(t, dtype=np.uint32)

    # initialise the current column with the starting points
    current_column = np.arange(round(m0), dtype=np.uint64)
    startpoints = np.arange(round(m0), dtype=np.uint64)

    # for each column
    for i in range(t):

        # get # cherry-picks for this column
        k_i = round(kis[i])

        # get indexes to trial
        pick_sample = np.array(random.sample(range(max_k_index), k_i), dtype=np.uint32)
        index_sample = (np.uint32(i) << np.uint32(k_bits)) | pick_sample

        # IN GPU

        # move current column to GPU
        # this doesn't work
        # current_column_gpu = cp.asarray(current_column)

        # find best seed for this column
        index = get_best_seed_distributed(current_column, index_sample)
        rf_indexes[i] = index

        # apply best reduction function on GPU 0
        with cp.cuda.Device(0):
            col_gpu = cp.asarray(current_column)

            # with the best seed, hash and reduce the current column

            reduced_points  = hash_reduce_gpu(col_gpu, int(index), N)
            del col_gpu

            # reduced_points = cp.empty(col_gpu.size, dtype=cp.uint64)
            # # hash and reduce
            # threads_per_block = 256                                                                                 # how many individual threads grouped together in a single block
            # blocks_per_grid = ceil(col_gpu.size / threads_per_block)                                     # how many blocks we need to cover all data
            # hash_reduce_kernel[blocks_per_grid, threads_per_block](col_gpu, index, reduced_points)       # launch the kernel with the specified grid and block dimensions



            # get the indicies of unique points to store the startpoints
            unique_points, sp_indicies = cp.unique(reduced_points, return_index=True)
            next_column = unique_points.get()
            selected_startpoints = startpoints[sp_indicies.get()]    
            
            del reduced_points, sp_indicies, unique_points
            cp.get_default_memory_pool().free_all_blocks()

        # next_column, sp_indicies = cp.unique(reduced_points, return_index=True)
        # selected_startpoints = startpoints[sp_indicies]

        # # update current column for next iteration
        # current_column = next_column.get()          # move back to CPU for next iteration
        # startpoints = selected_startpoints.get()    # move back to CPU for next iteration
        current_column = next_column
        startpoints = selected_startpoints


        # # clear GPU memory for next iteration
        # del current_column_gpu
        # del reduced_points
        # del next_column
        # del sp_indicies
        # del selected_startpoints
        # cp.get_default_memory_pool().free_all_blocks()

    # duration
    duration = perf_counter() - start_time
    print(f"Rainbow table built in {duration:.2f} seconds")


    return startpoints, current_column, rf_indexes


###############################################################################

if __name__ == "__main__":
    startpoints, final_column, rf_indexes = build(m0, t, kis)

    print("Length of final column:", len(final_column))
    print("startpoints & final column are equal length:", len(startpoints) == len(final_column))
    print("final column contains unique points:", len(final_column) == len(set(final_column)))
    print("rf_indexes are unique:", len(rf_indexes) == len(set(rf_indexes)))
