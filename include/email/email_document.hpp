#pragma once 

#include <string>
#include <vector>

namespace cortex::email
{
	// email struct define
	struct EmailDocument {
		std::string id;  // to identify a specific email
		std::string thread_id; // emails belong to the same thread / conversation


		std::string sender;
		std::vector<std::string> recipients;// to whome the message is recieved

		std::string subject;
		std::string body;

		std::string timestamp;

		std::vector<std::string> attachments; // the files that are send in the messages

		bool valid() const; // imp. for verfication of a valid email 

	};
}
