from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


# Initialize the LLM
model = ChatOpenAI()


# Create a prompt template with a dynamic {topic} variable
template1 = PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=['topic']
)

template2 = PromptTemplate(
    template="Summarize the given report in 5 bullet points. \n {report}",
    input_variables=['report']
)


# Converts the LLM's AIMessage response into a plain string
parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic': "Unemployement in India"})

print(result)