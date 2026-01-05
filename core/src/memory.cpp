#include "vectoria/memory.hpp"
#include <cstdlib>
#include <algorithm>

namespace vectoria {
namespace memory {

Arena::Arena(size_t block_size) : default_block_size_(block_size) {}

Arena::~Arena() {
    for (auto& block : blocks_) {
        std::free(block.data);
    }
}

void* Arena::allocate(size_t size, size_t alignment) {
    if (size == 0) return nullptr;

    for (auto& block : blocks_) {
        // Calculate padding for alignment
        uintptr_t current_addr = reinterpret_cast<uintptr_t>(block.data + block.used);
        size_t padding = (alignment - (current_addr % alignment)) % alignment;

        if (block.used + padding + size <= block.size) {
            void* ptr = block.data + block.used + padding;
            block.used += padding + size;
            return ptr;
        }
    }

    // No block has enough space, add a new one
    add_block(size + alignment);
    
    // We can safely assume the new block has enough space and is aligned at the start
    // (std::malloc usually returns memory aligned for any type)
    auto& block = blocks_.back();
    uintptr_t current_addr = reinterpret_cast<uintptr_t>(block.data);
    size_t padding = (alignment - (current_addr % alignment)) % alignment;
    
    void* ptr = block.data + padding;
    block.used = padding + size;
    return ptr;
}

void Arena::reset() {
    for (auto& block : blocks_) {
        block.used = 0;
    }
}

void Arena::add_block(size_t min_size) {
    size_t size = std::max(min_size, default_block_size_);
    void* data = std::malloc(size);
    if (!data) {
        throw std::bad_alloc();
    }
    blocks_.push_back({static_cast<uint8_t*>(data), size, 0});
}

} // namespace memory
} // namespace vectoria
