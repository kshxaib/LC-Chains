from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


# Initialize the LLM
model = ChatOpenAI()


# Create a prompt template with a dynamic {topic} variable
template = PromptTemplate(
    template="Generate 5 interesting facts about {topic}",
    input_variables=['topic']
)


# Converts the LLM's AIMessage response into a plain string
parser = StrOutputParser()


# LCEL chain:
# PromptTemplate → LLM → StrOutputParser
chain = template | model | parser


# Pass the value for {topic} and execute the chain
response = chain.invoke({'topic': 'Bugs'})


# Print the final parsed string
print(response)