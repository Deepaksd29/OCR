from dataclasses import dataclass
from dotenv import load_dotenv
import os
load_dotenv()




@dataclass
class LoadEnv:
    CHROMA_API_KEY:str = os.getenv("CHROMA_API_KEY")
    CHROMA_TENANT:str= os.getenv("CHROMA_TENANT")
    CHROMA_DATABASE:str=os.getenv("CHROMA_DATABASE")
    GOOGLE_API_KEY:str= os.getenv("GOOGLE_API_KEY")
    
    

env_data = LoadEnv()