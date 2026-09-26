#pragma once 
#include "email/email_document.hpp"

namespace cortex::email {
	class EmailCleaner {
	public:
		EmailDocument clean(const EmailDocument& email)const;
	};
}