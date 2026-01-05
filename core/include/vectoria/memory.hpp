#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>

namespace vectoria {
namespace memory {

/**
 * A simple arena-based allocator for deterministic memory management.
 * Memory is allocated in large blocks and handed out sequentially.
 * All memory is freed at once when the Arena is destroyed.
 */
class Arena {
public:
    explicit Arena(size_t block_size = 1024 * 1024); // Default 1MB blocks
    ~Arena();

    // Prevent copying
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    /**
     * Allocates a block of memory with the specified size and alignment.
     */
    void* allocate(size_t size, size_t alignment = alignof(std::max_align_t));

    /**
     * Resets the arena, making all memory available again.
     * Does not deallocate the underlying blocks from the system.
     */
    void reset();

private:
    struct Block {
        uint8_t* data;
        size_t size;
        size_t used;
    };

    size_t default_block_size_;
    std::vector<Block> blocks_;

    void add_block(size_t min_size);
};

} // namespace memory
} // namespace vectoria
