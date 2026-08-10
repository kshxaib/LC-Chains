from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser1 = StrOutputParser()

class FeedBack(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=FeedBack)


prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instructions}',
    input_variables=['feedback'],
    partial_variables={'format_instructions': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

# result = classifier_chain.invoke({'feedback': "This is a wonderful smart phone."})

prompt2 = PromptTemplate(
    template = "Write an appropriate response to this positive feedback \n {feedback}",
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template = "Write an appropriate response to this negative feedback \n {feedback}",
    input_variables=['feedback']
)


branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive' , prompt2 | model | parser1),
    (lambda x: x.sentiment == 'negative' , prompt3 | model | parser1),
    RunnableLambda(lambda x: "Neutral Feedback")
)

chain = classifier_chain | branch_chain

result = chain.invoke({'feedback' : 'It was a great experience'})

print(result)