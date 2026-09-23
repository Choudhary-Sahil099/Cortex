#pragma once

#include "index/hnsw.hpp"

#include <string>

namespace cortex::persistence {

    class Serializer {
        public:
            // saving api
            static void save(
                const index::HNSWIndex& index,
                const std::string& path
            );


            // laod api
            static index::HNSWIndex load(
                const std::string& path
            );

        private:
            static constexpr std::uint64_t MAGIC = 0x434F525445585F31; // hexa marker
            static constexpr std::uint32_t VERSION = 1;// format teller
    };

}