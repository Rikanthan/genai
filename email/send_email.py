from sib_api_v3_sdk import ApiClient, Configuration
from sib_api_v3_sdk.api.transactional_emails_api import TransactionalEmailsApi
from sib_api_v3_sdk.models import SendSmtpEmail
from sib_api_v3_sdk.rest import ApiException

# Configure API key
configuration = Configuration()
configuration.api_key['api-key'] = ''

# Create API instance
api_instance = TransactionalEmailsApi(ApiClient(configuration))

# Compose the email
email = SendSmtpEmail(
    to=[{"email": "kingnotfound1@gmail.com", "name": "Rikanthan"}],
    sender={"name": "My From Name", "email": "rikanthanricky@gmail.com"},
    subject="My subject",
    html_content="<html><body><h1>Congratulations!</h1><p>You successfully sent this example email via the Brevo API.</p></body></html>"
)

# Send the email
try:
    response = api_instance.send_transac_email(email)
    print("Email sent successfully. Message ID:", response.message_id)
except ApiException as e:
    print("Exception when calling send_transac_email: %s\n" % e)
