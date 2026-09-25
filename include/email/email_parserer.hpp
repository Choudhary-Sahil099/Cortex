#pragma once
#include "email/email_document.hpp"

#include <string>
namespace cortex::email {
		class EmailParser {
			public:
				EmailDocument parse(const std::string& raw_email) const; // document parsew
		};
}