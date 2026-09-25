#include "email/email_document.hpp"

namespace cortex::email {

    // email validation v1
    bool EmailDocument::valid() const
    {
        return !id.empty()
            && !thread_id.empty()
            && !sender.empty()
            && !body.empty();
    }

}