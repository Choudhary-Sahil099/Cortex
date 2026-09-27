#pragma once
#include "email/email_document.hpp"
#include "email/email_chunk.hpp"

#include <cstddef>
#include<vector>

namespace cortex::email {
	class EmailChunker {
		public:
			explicit EmailChunker(size_t chunk_size = 500); //The size is a fixed size parameter for the intial phase only and will be updated in the future
			std::vector<emailChunk> chunk(const EmailDocument& email) const;

		private:
			std::size_t chunk_size_;
	};
} 