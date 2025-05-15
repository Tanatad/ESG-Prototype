from langchain_core.rate_limiters import InMemoryRateLimiter

max_tokens_per_request = 5000

# test file: รายงานนโยบายและเอกสารด้าน ESG.pdf
# Free tier: test file ~ 2 mins
max_tpm = 1000000  
max_rpm = 15

# Paid tier 1: test file ~ 35 secs
# max_tpm = 4000000
# max_rpm = 2000

tokens_per_second = max_tpm / 60  # Tokens per second
requests_per_second = max_rpm / 60  # Requests per second

llm_rate_limiter = InMemoryRateLimiter(
    requests_per_second=requests_per_second,
    check_every_n_seconds=0.1,  # You can adjust this interval if needed
    max_bucket_size=max_tpm // max_tokens_per_request  # Maximum number of tokens allowed in the bucket
)