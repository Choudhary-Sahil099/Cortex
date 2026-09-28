#pragma once 
#include <string>
#include <vector>

namespace cortex::embedding {

	// for finding a better dimension size we have not hardcoded it yet and the size will be finalized after the real testing process
	class Embedder {
		public:
			virtual ~Embedder() = default;
			virtual std::vector<float> embed(const std::string& text) const = 0;
			virtual std::size_t dimension() const = 0;
	};
}