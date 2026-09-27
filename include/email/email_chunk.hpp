#pragma once

#include <cstddef>
#include <string>

// defining the struct of the email chunk

namespace cortex::email {
	struct emailChunk {
		std::string id;// new id
		std::string email_id; // have the original email_id;
		std::string thread_id;
		std::string text;
		std::size_t index; // embedded content
	};
}