import openai
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, wait_exponential, stop_after_attempt

class LLMClient:
    def __init__(self, api_key: str):
        self.llm_client = OpenAI(
            api_key = api_key,
            base_url="https://api.groq.com/openai/v1"
        )
    
    @retry(
        retry=retry_if_exception_type(openai.RateLimitError),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        stop=stop_after_attempt(6),
        before_sleep=lambda retry_state: print(f"Rate limited. Retrying in {retry_state.next_action.sleep} seconds...")
    )
    def prompt_llm(self, prompt:str, llm_model: str = 'openai/gpt-oss-120b'):
        response = self.llm_client.chat.completions.create(
            model=llm_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return {
                'response': response,
                'model': llm_model 
            }

# import openai
# from openai import OpenAI
# from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# class LLMBaseClient:
#     def __init__(self, base_url: str, api_key: str):
#         self.llm_client = OpenAI(
#             api_key = api_key,
#             base_url=base_url
#         )
    
#     @retry(
#         retry=retry_if_exception_type(openai.RateLimitError),
#         wait=wait_exponential(multiplier=1, min=2, max=30),
#         stop=stop_after_attempt(5)
#     )
#     def prompt_llm(self, prompt:str, llm_model: str = 'gemini-3.1-flash-lite-preview'):
#         response = self.llm_client.chat.completions.create(
#             model=llm_model,
#             messages=[
#                 {
#                     "role": "user",
#                     "content": prompt
#                 }
#             ]
#         )
#         return response

# class LLMSuperClient:
#     SUPPORTED_CONFIGS = {
#         'google': {
#             'base_url': 'https://generativelanguage.googleapis.com/v1beta/openai/',
#             'supported_models': {
#                 'gemini-3.1-flash-lite-preview': {
#                     'rpm': 15,
#                     'tpm': 250000,
#                     'rpd': 500,
#                     'tpd': None
#                 }
#             }
#         },
#         # 'openai': {
#         #     'base_url': 'https://api.openai.com/v1',
#         #     'supported_models': ['gpt-4o', 'gpt-4-turbo', 'gpt-3.5-turbo']
#         # },
#         'groq': {
#             'base_url': 'https://api.groq.com/openai/v1',
#             'supported_models': {
#                 'llama-3.1-8b-instant': {
#                     'rpm': 30,
#                     'rpd': 14400,
#                     'tpm': 6000,
#                     'tpd': 500000
#                 },
#                 'llama-3.3-70b-versatile': {
#                     'rpm': 30,
#                     'rpd': 1000,
#                     'tpm': 12000,
#                     'tpd': 100000
#                 },
#                 'openai/gpt-oss-120b': {
#                     'rpm': 30,
#                     'rpd': 1000,
#                     'tpm': 8000,
#                     'tpd': 200000
#                 },
#                 'openai/gpt-oss-20b': {
#                     'rpm': 30,
#                     'rpd': 1000,
#                     'tpm': 8000,
#                     'tpd': 200000
#                 },
#                 'qwen/qwen3-32b': {
#                     'rpm': 60,
#                     'rpd': 1000,
#                     'tpm': 6000,
#                     'tpd': 500000
#                 }
#             }
#         }
#     }

#     def __init__(self, client_dict: list[dict[str, str]]):
#         clients:list[LLMBaseClient] = []
#         for client_config in client_dict:
#             clients.append(LLMBaseClient(**client_config))